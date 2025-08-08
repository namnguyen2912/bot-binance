import os
import time
import numpy as np
import pandas as pd
from binance.um_futures import UMFutures
from binance.error import ClientError
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from datetime import datetime

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("❌ Vui lòng đặt biến môi trường BINANCE_API_KEY và BINANCE_API_SECRET")

client = UMFutures(key=API_KEY, secret=API_SECRET)

TOTAL_CAPITAL = 1000
TRADE_PERCENTAGE = 0.05  # 5% mỗi lệnh
POSITIONS = {}  # Lưu vị thế đang mở
symbol_step_size = {}  # Cache step size

def get_step_size(symbol):
    if symbol in symbol_step_size:
        return symbol_step_size[symbol]
    try:
        info = client.exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'LOT_SIZE':
                        step = float(f['stepSize'])
                        symbol_step_size[symbol] = step
                        return step
    except Exception as e:
        print(f"⚠️ Lỗi lấy stepSize của {symbol}: {e}")
    return 0.001

def round_qty(symbol, qty):
    step = get_step_size(symbol)
    precision = int(round(-np.log10(step)))
    return round(qty, precision)

def fetch_klines(symbol, interval='1m', limit=100):
    try:
        klines = client.klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'])
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        return df
    except ClientError as e:
        print(f"❌ Lỗi fetch klines {symbol}: {e}")
        return None

def train_model(df):
    df['return'] = df['close'].pct_change().fillna(0)
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    df['target'] = (df['future_return'] > 0.001).astype(int)

    X = df[['close', 'volume']].copy()
    y = df['target']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LGBMClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def get_prediction(model, df):
    latest = df[['close', 'volume']].iloc[-1:]
    return model.predict(latest)[0]

def get_top_symbols(limit=5):
    tickers = client.ticker_24hr_price_change()
    df = pd.DataFrame(tickers)
    df['quoteVolume'] = pd.to_numeric(df['quoteVolume'], errors='coerce')
    df = df[df['symbol'].str.endswith('USDT') & ~df['symbol'].str.contains('1000')]
    df = df.sort_values('quoteVolume', ascending=False).head(limit)
    return df['symbol'].tolist()

def place_order(symbol, side, qty):
    try:
        response = client.new_order(symbol=symbol, side=side, type='MARKET', quantity=qty)
        print(f"✅ Đặt lệnh {side} {symbol} thành công: {response}")
        return response
    except ClientError as e:
        print(f"❌ Lỗi đặt lệnh {symbol}-{side}: {e.message}")
        return None

def check_positions():
    global POSITIONS, TOTAL_CAPITAL
    for symbol, pos in list(POSITIONS.items()):
        price = float(client.ticker_price(symbol)['price'])
        entry = pos['entry']
        qty = pos['qty']
        pnl = (price - entry) / entry

        if pnl >= 0.03 or pnl <= -0.015:
            print(f"📤 Bán {symbol} | Entry: {entry} | Now: {price} | PnL: {pnl*100:.2f}%")
            place_order(symbol, 'SELL', qty)
            usdt_returned = qty * price
            TOTAL_CAPITAL += usdt_returned
            del POSITIONS[symbol]

def trade_loop():
    global TOTAL_CAPITAL
    while True:
        check_positions()
        symbols = get_top_symbols()
        for symbol in symbols:
            df = fetch_klines(symbol)
            if df is None or len(df) < 20:
                continue
            model = train_model(df)
            prediction = get_prediction(model, df)
            price = df['close'].iloc[-1]
            print(f"\n📈 {symbol} | Giá: {price:.2f} | AI: {prediction}")

            if prediction == 1 and symbol not in POSITIONS:
                usdt_amount = TOTAL_CAPITAL * TRADE_PERCENTAGE
                qty = round_qty(symbol, usdt_amount / price)
                if qty <= 0:
                    print(f"⚠️ Số lượng {qty} không hợp lệ cho {symbol}")
                    continue
                response = place_order(symbol, 'BUY', qty)
                if response:
                    POSITIONS[symbol] = {'entry': price, 'qty': qty}
                    TOTAL_CAPITAL -= usdt_amount
        time.sleep(60)

if __name__ == '__main__':
    print("🤖 Bắt đầu bot Futures AI...")
    trade_loop()
