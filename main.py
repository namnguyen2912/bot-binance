import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from binance.client import Client
from binance.enums import *

# ==== ENV ====
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# ==== Client testnet ====
client = Client(API_KEY, API_SECRET)
client.API_URL = 'https://testnet.binance.vision/api'

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1HOUR

# ==== Fetch historical OHLCV ====
def fetch_ohlcv(symbol, interval, lookback_days=365):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    print(f"📊 Lấy dữ liệu từ {start_time.date()} đến {end_time.date()}...")

    klines = client.get_historical_klines(symbol, interval, start_time.strftime("%d %b %Y %H:%M:%S"))
    df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                                       'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                                       'Taker_buy_base_vol', 'Taker_buy_quote_vol', 'Ignore'])
    df = df[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]
    df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.astype(float, errors='ignore')
    return df.dropna()

# ==== Indicators & Features ====
def rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def create_features(data):
    df_feat = data.copy()
    df_feat['return'] = df_feat['close'].pct_change()
    df_feat['ema5'] = df_feat['close'].ewm(span=5).mean()
    df_feat['ema10'] = df_feat['close'].ewm(span=10).mean()
    df_feat['ema20'] = df_feat['close'].ewm(span=20).mean()
    df_feat['ema_cross'] = np.where(df_feat['ema5'] > df_feat['ema10'], 1, -1)
    df_feat['rsi'] = rsi(df_feat['close'], 14)
    df_feat['future_return'] = df_feat['close'].shift(-5) / df_feat['close'] - 1
    df_feat['target'] = np.where(df_feat['future_return'] > 0.002, 1,
                                 np.where(df_feat['future_return'] < -0.002, -1, 0))
    return df_feat.dropna()

# ==== Train AI model ====
def train_model(df_feat):
    feature_cols = ['return', 'ema5', 'ema10', 'ema20', 'ema_cross', 'rsi']
    X = df_feat[feature_cols]
    y = df_feat['target']

    if len(X) < 100 or y.nunique() < 2:
        print("⚠️ Dữ liệu huấn luyện không đủ hoặc thiếu nhãn phân loại. Bỏ qua huấn luyện.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    best_params = {'n_estimators': 133, 'max_depth': 7, 'learning_rate': 0.05}
    model = LGBMClassifier(**best_params)
    model.fit(X_train, y_train)
    print("=== BÁO CÁO PHÂN LOẠI AI ===")
    print(classification_report(y_test, model.predict(X_test)))
    return model

# ==== Giao dịch thực ====
def get_balance(asset):
    try:
        balance = client.get_asset_balance(asset=asset)
        free_amount = float(balance['free']) if balance else 0
        print(f"💰 Số dư {asset}: {free_amount}")
        return free_amount
    except Exception as e:
        print(f"⚠️ Không thể lấy số dư {asset}: {e}")
        return 0

def place_order(side, quantity):
    try:
        order = client.create_order(
            symbol=symbol,
            side=side,
            type=ORDER_TYPE_MARKET,
            quantity=quantity
        )
        print(f"🟢 Đặt lệnh {side} thành công: {order}")
        return order
    except Exception as e:
        print(f"❌ Lỗi đặt lệnh {side}: {e}")
        return None

def calculate_quantity(price, usdt_balance, fixed_amount=1000):
    amount = min(usdt_balance, fixed_amount)
    return round(amount / price, 6)

# ==== Bot logic ====
def run_bot():
    df = fetch_ohlcv(symbol, interval)
    df_feat = create_features(df)
    model = train_model(df_feat)
    if model is None:
        return

    latest = df_feat.iloc[[-1]]  # DataFrame
    X_live = latest[['return', 'ema5', 'ema10', 'ema20', 'ema_cross', 'rsi']]
    pred = model.predict(X_live)[0]

    price = latest['close'].values[0]
    print(f"📈 Giá hiện tại: {price:.2f} | Tín hiệu AI: {pred}")

    usdt = get_balance("USDT")
    btc = get_balance("BTC")

    if pred == 1:
        print("✅ AI tín hiệu MUA")
        if usdt >= 10:
            qty = calculate_quantity(price, usdt)
            print(f"👉 Đặt mua ~{qty} BTC (tương đương ~1000 USDT)")
            place_order(SIDE_BUY, qty)
    elif pred == -1:
        print("✅ AI tín hiệu BÁN")
        if btc >= 0.0001:
            print(f"👉 Đặt bán toàn bộ {btc} BTC")
            place_order(SIDE_SELL, round(btc, 6))
    else:
        print("⏸️ Không có tín hiệu rõ ràng từ AI")

# ==== Loop chính ====
if __name__ == "__main__":
    print("🚀 AI Binance Bot khởi động...")
    while True:
        try:
            run_bot()
            print("🕐 Đợi 60s...\n")
            time.sleep(60)
        except Exception as e:
            print(f"🔥 Lỗi: {e}")
            time.sleep(60)
