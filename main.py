import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta

# 🔐 Điền API Testnet
API_KEY = 'YOUR_TESTNET_API_KEY'
API_SECRET = 'YOUR_TESTNET_API_SECRET'

symbol = 'BTC/USDT'
timeframe = '15m'
short_window = 5
long_window = 20
amount = 0.001  # Lượng BTC muốn giao dịch mỗi lần

# ✅ Khởi tạo Binance testnet client
binance = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})
binance.set_sandbox_mode(True)

# ✅ Hàm lấy dữ liệu nến từ Binance
def fetch_binance_ohlcv(symbol, timeframe='15m', since_days=5):
    since = binance.parse8601((datetime.utcnow() - timedelta(days=since_days)).isoformat())
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# ✅ Hàm tính tín hiệu MA
def generate_signals(df, short_window=5, long_window=20):
    df = df.copy()
    df['short_ma'] = df['close'].rolling(window=short_window).mean()
    df['long_ma'] = df['close'].rolling(window=long_window).mean()
    df['position'] = np.where(df['short_ma'] > df['long_ma'], 1, 0)
    df['signal'] = df['position'].diff()
    return df

# ✅ Đặt lệnh market MUA/BÁN
def place_order(signal):
    try:
        if signal == 1:
            order = binance.create_market_buy_order(symbol, amount)
            print(f"🟢 Đã đặt lệnh MUA: {order}")
        elif signal == -1:
            order = binance.create_market_sell_order(symbol, amount)
            print(f"🔴 Đã đặt lệnh BÁN: {order}")
    except Exception as e:
        print(f"❌ Lỗi khi đặt lệnh: {e}")

# ✅ Vòng lặp bot trading
def trading_loop():
    print(f"🚀 Bắt đầu bot MA trên Testnet Binance - Symbol: {symbol}")
    while True:
        df = fetch_binance_ohlcv(symbol, timeframe, since_days=5)
        df = generate_signals(df, short_window, long_window)

        last_signal = df['signal'].iloc[-1]
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Tín hiệu: {last_signal}")

        if last_signal == 1:
            place_order(1)
        elif last_signal == -1:
            place_order(-1)
        else:
            print("🔄 Không có tín hiệu mới")

        # Chờ 15 phút
        time.sleep(15 * 60)

# 👉 Gọi hàm để chạy bot
# trading_loop()  # Bỏ comment dòng này để chạy bot
