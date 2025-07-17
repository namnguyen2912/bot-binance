import time
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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
capital = 1000  # Giả định số vốn ban đầu để tính khối lượng trade (không dùng nếu lấy từ số dư thực)

# ==== Lấy dữ liệu nến ====
def fetch_ohlcv(symbol, interval, lookback_days=365):
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=lookback_days)
    print(f"📊 Lấy dữ liệu từ {start_time.date()} đến {end_time.date()}...")

    klines = client.get_historical_klines(symbol, interval, start_time.strftime("%d %b %Y %H:%M:%S"))
    df = pd.DataFrame(klines, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume',
                                       'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                                       'Taker_buy_base_vol', 'Taker_buy_quote_vol', 'Ignore'])

    df['Close'] = df['Close'].astype(float)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df[['timestamp', 'Close']]

# ==== Chiến lược MA ====
def apply_strategy(df, short_window=10, long_window=20):
    df['short_ma'] = df['Close'].rolling(window=short_window).mean()
    df['long_ma'] = df['Close'].rolling(window=long_window).mean()
    df['signal'] = 0
    df.loc[short_window:, 'signal'] = np.where(df['short_ma'][short_window:] > df['long_ma'][short_window:], 1, 0)
    df['position'] = df['signal'].diff()
    return df

# ==== Lệnh giao dịch ====
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

# ==== Kiểm tra số dư USDT và BTC ====
def get_balance(asset):
    try:
        balance = client.get_asset_balance(asset=asset)
        free_amount = float(balance['free']) if balance else 0
        print(f"💰 Số dư {asset}: {free_amount}")
        return free_amount
    except Exception as e:
        print(f"⚠️ Không thể lấy số dư {asset}: {e}")
        return 0

# ==== Tính khối lượng trade ====
def calculate_quantity(price, usdt_balance):
    quantity = round(usdt_balance / price, 6)
    return quantity

# ==== BOT ====
def run_bot():
    df = fetch_ohlcv(symbol, interval)
    df = apply_strategy(df)

    last_signal = df['position'].iloc[-1]
    last_price = df['Close'].iloc[-1]

    print(f"📈 Giá hiện tại: {last_price:.2f}")

    usdt_balance = get_balance("USDT")
    btc_balance = get_balance("BTC")
    quantity = calculate_quantity(last_price, usdt_balance)

    if last_signal == 1:
        print("✅ Tín hiệu MUA")
        if usdt_balance >= 10:  # Binance yêu cầu số lệnh > 10 USDT
            place_order(SIDE_BUY, quantity)
        else:
            print("⚠️ Không đủ USDT để mua.")
    elif last_signal == -1:
        print("✅ Tín hiệu BÁN")
        if btc_balance >= 0.0001:
            place_order(SIDE_SELL, round(btc_balance, 6))
        else:
            print("⚠️ Không đủ BTC để bán.")
    else:
        print("⏸️ Không có tín hiệu giao dịch.")

# ==== Loop chính ====
if __name__ == "__main__":
    print("🚀 Khởi động bot giao dịch Binance Testnet...")
    while True:
        try:
            run_bot()
            print("🕐 Đợi 60s...\n")
            time.sleep(60)
        except Exception as e:
            print(f"🔥 Lỗi: {e}")
            time.sleep(60)
