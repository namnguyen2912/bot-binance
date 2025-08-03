# === AI Binance Futures Bot (testnet) ===
# Kết hợp thuật toán AI + giao dịch Futures với quản lý vốn 5%, chốt lời 3%, cắt lỗ 1.5%

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

# ==== ENV ====
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")
client = UMFutures(key=API_KEY, secret=API_SECRET, base_url="https://testnet.binancefuture.com")

symbol = "BTCUSDT"
interval = '1h'

CAPITAL = 1000.0               # Tổng vốn (giả định hoặc quản lý nội bộ)
ORDER_PERCENT = 0.05           # Mỗi lệnh tối đa 5%
TP_PCT = 0.03                  # Take profit
SL_PCT = 0.015                 # Stop loss

open_positions = []            # Theo dõi vị thế mở: [{'qty':..., 'entry':...}]

# ==== Dữ liệu giá ====
def fetch_ohlcv(symbol, interval, lookback_days=365):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    klines = client.klines(symbol=symbol, interval=interval, startTime=int(start_time.timestamp()*1000))
    df = pd.DataFrame(klines, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time",
                                       "qav", "num_trades", "tbbav", "tbqav", "ignore"])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df = df.astype(float, errors='ignore')
    return df.dropna()

# ==== Chỉ báo & đặc trưng ====
def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def create_features(df):
    df_feat = df.copy()
    df_feat['return'] = df_feat['close'].pct_change()
    df_feat['ema5'] = df_feat['close'].ewm(span=5).mean()
    df_feat['ema10'] = df_feat['close'].ewm(span=10).mean()
    df_feat['ema20'] = df_feat['close'].ewm(span=20).mean()
    df_feat['ema_cross'] = np.where(df_feat['ema5'] > df_feat['ema10'], 1, -1)
    df_feat['rsi'] = rsi(df_feat['close'])
    df_feat['future_return'] = df_feat['close'].shift(-5) / df_feat['close'] - 1
    df_feat['target'] = np.where(df_feat['future_return'] > 0.002, 1,
                                 np.where(df_feat['future_return'] < -0.002, -1, 0))
    return df_feat.dropna()

# ==== Huấn luyện AI ====
def train_model(df_feat):
    X = df_feat[['return', 'ema5', 'ema10', 'ema20', 'ema_cross', 'rsi']]
    y = df_feat['target']

    if len(X) < 100 or y.nunique() < 2:
        print("⚠️ Dữ liệu không đủ.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LGBMClassifier(n_estimators=133, max_depth=7, learning_rate=0.05)
    model.fit(X_train, y_train)
    print("=== AI MODEL ===")
    print(classification_report(y_test, model.predict(X_test)))
    return model

# ==== Đặt lệnh ====
def place_order(side, qty):
    try:
        return client.new_order(symbol=symbol, side=side, type="MARKET", quantity=qty)
    except ClientError as e:
        print(f"❌ Lỗi đặt lệnh {side}: {e.error_message}")
        return None

def get_price():
    ticker = client.ticker_price(symbol=symbol)
    return float(ticker['price'])

def round_qty(value):
    return round(value, 3)

# ==== Kiểm tra vị thế để đóng ====
def check_close_positions(price):
    global open_positions
    to_close = []
    for i, pos in enumerate(open_positions):
        entry = pos['entry']
        qty = pos['qty']
        pnl = (price - entry) / entry
        if pnl >= TP_PCT:
            print(f"🚀 Chốt lời {qty} BTC, vào {entry} ra {price} ({pnl*100:.2f}%)")
            place_order('SELL', qty)
            to_close.append(i)
        elif pnl <= -SL_PCT:
            print(f"🔻 Cắt lỗ {qty} BTC, vào {entry} ra {price} ({pnl*100:.2f}%)")
            place_order('SELL', qty)
            to_close.append(i)
    for i in reversed(to_close):
        open_positions.pop(i)

# ==== Bot chính ====
def run_bot():
    global open_positions
    df = fetch_ohlcv(symbol, interval)
    df_feat = create_features(df)
    model = train_model(df_feat)
    if model is None:
        return

    latest = df_feat.iloc[[-1]]
    X_live = latest[['return', 'ema5', 'ema10', 'ema20', 'ema_cross', 'rsi']]
    pred = model.predict(X_live)[0]
    price = latest['close'].values[0]

    print(f"\n📉 Giá hiện tại: {price:.2f} | Tín hiệu AI: {pred}")
    check_close_positions(price)

    if pred == 1:
        print("✅ AI báo MUA")
        usdt_amount = CAPITAL * ORDER_PERCENT
        qty = round_qty(usdt_amount / price)
        if qty > 0:
            res = place_order('BUY', qty)
            if res:
                open_positions.append({'qty': qty, 'entry': price})
    elif pred == -1:
        print("✅ AI báo BÁN")
        total_qty = round_qty(sum(p['qty'] for p in open_positions))
        if total_qty > 0:
            res = place_order('SELL', total_qty)
            if res:
                open_positions.clear()
    else:
        print("⏸️ AI không chắc chắn.")

# ==== Loop ====
if __name__ == '__main__':
    print("🚀 Khởi động bot Futures...")
    while True:
        try:
            run_bot()
            print("⏱️ Đợi 60s...")
            time.sleep(60)
        except Exception as e:
            print(f"🔥 Lỗi: {e}")
            time.sleep(60)
