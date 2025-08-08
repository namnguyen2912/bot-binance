import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from binance.um_futures import UMFutures
from binance.error import ClientError

# === ENV ===
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# === Client Futures Testnet ===
client = UMFutures(key=API_KEY, secret=API_SECRET, base_url="https://testnet.binancefuture.com")

INTERVAL = '1h'
ORDER_PERCENT = 0.05
TP_PCT = 0.03
SL_PCT = 0.015

# === Vị thế mở dạng {symbol: {qty, entry}} ===
open_positions = {}

# === Top symbols (volume cao) ===
def get_top_symbols(limit=5):
    try:
        res = client.ticker_24hr_price_change()
        df = pd.DataFrame(res)
        df['volume'] = df['quoteVolume'].astype(float)
        df = df[df['symbol'].str.endswith("USDT")]
        top = df.sort_values(by='volume', ascending=False).head(limit)
        return top['symbol'].tolist()
    except Exception as e:
        print(f"⚠️ Lỗi get_top_symbols: {e}")
        return ["BTCUSDT"]

# === Vốn thực tế ===
def get_total_capital():
    try:
        balance = client.balance()
        usdt = next((float(b['balance']) for b in balance if b['asset'] == 'USDT'), 0.0)
        print(f"\n💰 USDT hiện có: {usdt}")
        return usdt
    except Exception as e:
        print(f"⚠️ Lỗi lấy vốn: {e}")
        return 0.0

# === Lấy dữ liệu giá ===
def fetch_ohlcv(symbol, interval, lookback_days=90):
    end = datetime.utcnow()
    start = end - timedelta(days=lookback_days)
    try:
        klines = client.klines(symbol=symbol, interval=interval, startTime=int(start.timestamp()*1000))
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "close_time",
            "qav", "num_trades", "tbbav", "tbqav", "ignore"
        ])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df = df.astype(float, errors='ignore')
        return df.dropna()
    except Exception as e:
        print(f"⚠️ Lỗi fetch_ohlcv {symbol}: {e}")
        return pd.DataFrame()

# === Các chỉ báo ===
def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, window=20, num_std=2):
    ma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, lower

def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = low.diff().abs()
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# === Tạo đặc trưng ===
def create_features(df):
    df_feat = df.copy()
    df_feat['return'] = df_feat['close'].pct_change()
    df_feat['ema10'] = df_feat['close'].ewm(span=10).mean()
    df_feat['ema20'] = df_feat['close'].ewm(span=20).mean()
    df_feat['ema_cross'] = np.where(df_feat['ema10'] > df_feat['ema20'], 1, -1)
    df_feat['rsi'] = rsi(df_feat['close'])
    df_feat['upper'], df_feat['lower'] = bollinger_bands(df_feat['close'])
    df_feat['adx'] = adx(df_feat['high'], df_feat['low'], df_feat['close'])

    df_feat['future_return'] = df_feat['close'].shift(-5) / df_feat['close'] - 1
    df_feat['target'] = np.where(df_feat['future_return'] > 0.002, 1,
                                 np.where(df_feat['future_return'] < -0.002, -1, 0))
    return df_feat.dropna()

# === Huấn luyện AI ===
def train_model(df_feat):
    X = df_feat[['return', 'ema10', 'ema20', 'ema_cross', 'rsi', 'upper', 'lower', 'adx']]
    y = df_feat['target']
    if len(X) < 100 or y.nunique() < 2:
        return None
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model

# === Đặt lệnh ===
def place_order(symbol, side, qty):
    try:
        return client.new_order(symbol=symbol, side=side, type="MARKET", quantity=qty)
    except ClientError as e:
        print(f"❌ Lỗi đặt lệnh {symbol}-{side}: {e.error_message}")
        return None

def round_qty(value):
    return round(value, 3)

# === Kiểm tra đóng lệnh ===
def check_close(symbol, price):
    if symbol not in open_positions:
        return
    entry = open_positions[symbol]['entry']
    qty = open_positions[symbol]['qty']
    pnl = (price - entry) / entry
    if pnl >= TP_PCT:
        print(f"🚀 TP {symbol} (+{pnl*100:.2f}%)")
        place_order(symbol, "SELL", qty)
        del open_positions[symbol]
    elif pnl <= -SL_PCT:
        print(f"🔻 SL {symbol} ({pnl*100:.2f}%)")
        place_order(symbol, "SELL", qty)
        del open_positions[symbol]

# === BOT chính ===
def run_bot():
    global open_positions
    total_cap = get_total_capital()
    if total_cap < 10:
        print("⚠️ Vốn thấp, dừng bot.")
        return

    symbols = get_top_symbols(limit=5)
    for sym in symbols:
        df = fetch_ohlcv(sym, INTERVAL)
        if df.empty:
            continue

        df_feat = create_features(df)
        model = train_model(df_feat)
        if model is None:
            continue

        latest = df_feat.iloc[[-1]]
        X_live = latest[['return', 'ema10', 'ema20', 'ema_cross', 'rsi', 'upper', 'lower', 'adx']]
        pred = model.predict(X_live)[0]
        price = latest['close'].values[0]

        print(f"\n📈 {sym} | Giá: {price:.2f} | AI: {pred}")
        check_close(sym, price)

        if pred == 1 and sym not in open_positions:
            qty = round_qty((total_cap * ORDER_PERCENT) / price)
            if qty > 0:
                res = place_order(sym, "BUY", qty)
                if res:
                    open_positions[sym] = {'qty': qty, 'entry': price}
                    print(f"✅ Mua {sym} @ {price} với {qty}")
        elif pred == -1 and sym in open_positions:
            qty = open_positions[sym]['qty']
            res = place_order(sym, "SELL", qty)
            if res:
                del open_positions[sym]
                print(f"✅ Bán {sym} @ {price} với {qty}")
        else:
            print("⏸️ Không hành động.")

# === LOOP ===
if __name__ == '__main__':
    print("🤖 Khởi động bot AI Futures đa coin...")
    while True:
        try:
            run_bot()
            print("🕒 Đợi 60s...\n")
            time.sleep(60)
        except Exception as e:
            print(f"🔥 Lỗi chính: {e}")
            time.sleep(60)
