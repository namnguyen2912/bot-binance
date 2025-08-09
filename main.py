import os
import time
import numpy as np
import pandas as pd
from binance.um_futures import UMFutures
from binance.error import ClientError
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("❌ Vui lòng đặt biến môi trường BINANCE_API_KEY và BINANCE_API_SECRET")

client = UMFutures(key=API_KEY, secret=API_SECRET)

TOTAL_CAPITAL = 1000
TRADE_PERCENTAGE = 0.05
POSITIONS = {}
symbol_info_cache = {}

def get_symbol_info(symbol):
    if symbol in symbol_info_cache:
        return symbol_info_cache[symbol]
    try:
        info = client.exchange_info()
        for s in info['symbols']:
            if s['symbol'] == symbol:
                symbol_info_cache[symbol] = s
                return s
    except Exception as e:
        print(f"⚠️ Lỗi lấy thông tin {symbol}: {e}")
    return None

def adjust_qty(symbol, qty, price):
    info = get_symbol_info(symbol)
    if not info:
        return qty
    step_size = min_qty = min_notional = 0.0
    for f in info['filters']:
        if f['filterType'] == 'LOT_SIZE':
            step_size = float(f['stepSize'])
            min_qty = float(f['minQty'])
        elif f['filterType'] == 'MIN_NOTIONAL':
            min_notional = float(f['notional'])
    qty = max(qty, min_qty)
    if price * qty < min_notional:
        qty = min_notional / price
    precision = int(round(-np.log10(step_size))) if step_size > 0 else 3
    return round(qty, precision)

def fetch_klines(symbol, interval='15m', limit=200):
    try:
        klines = client.klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'])
        df = df.astype({'close': float, 'volume': float})
        return df
    except ClientError as e:
        print(f"❌ Lỗi fetch klines {symbol}: {e}")
        return None

def add_features(df):
    df['return'] = df['close'].pct_change().fillna(0)
    df['ema10'] = df['close'].ewm(span=10).mean()
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['rsi'] = compute_rsi(df['close'])
    df['future_return'] = df['close'].shift(-3) / df['close'] - 1
    df['target'] = (df['future_return'] > 0.002).astype(int)
    return df.dropna()

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model(df):
    features = ['return', 'ema10', 'ema20', 'rsi']
    X = df[features]
    y = df['target']
    if len(X) < 50 or y.nunique() < 2:
        return None
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model

def get_prediction(model, df):
    if model is None:
        return 0
    latest = df[['return', 'ema10', 'ema20', 'rsi']].iloc[-1:]
    return model.predict(latest)[0]

def get_top_symbols(limit=5):
    tickers = client.ticker_24hr_price_change()
    df = pd.DataFrame(tickers)
    df['quoteVolume'] = pd.to_numeric(df['quoteVolume'], errors='coerce')
    df = df[df['symbol'].str.endswith('USDT') & ~df['symbol'].str.contains('1000')]
    return df.sort_values('quoteVolume', ascending=False)['symbol'].head(limit).tolist()

def place_order(symbol, side, qty):
    try:
        response = client.new_order(symbol=symbol, side=side, type='MARKET', quantity=qty)
        print(f"✅ Lệnh {side} {symbol} thành công: {response}")
        return response
    except ClientError as e:
        print(f"❌ Lỗi lệnh {side} {symbol}: {e.message}")
        return None

def check_positions():
    global POSITIONS, TOTAL_CAPITAL
    for symbol, pos in list(POSITIONS.items()):
        price = float(client.ticker_price(symbol)['price'])
        pnl = (price - pos['entry']) / pos['entry']
        if pnl >= 0.03 or pnl <= -0.015:
            print(f"📤 Đóng {symbol} | Entry: {pos['entry']} | Now: {price} | PnL: {pnl*100:.2f}%")
            if place_order(symbol, 'SELL', pos['qty']):
                TOTAL_CAPITAL += pos['qty'] * price
                del POSITIONS[symbol]

def trade_loop():
    global TOTAL_CAPITAL
    while True:
        check_positions()
        for symbol in get_top_symbols():
            df = fetch_klines(symbol)
            if df is None or len(df) < 30:
                continue
            df_feat = add_features(df)
            model = train_model(df_feat)
            pred = get_prediction(model, df_feat)
            price = df_feat['close'].iloc[-1]
            print(f"\n📈 {symbol} | Giá: {price:.2f} | AI: {pred}")
            if pred == 1 and symbol not in POSITIONS:
                usdt_amount = TOTAL_CAPITAL * TRADE_PERCENTAGE
                qty = adjust_qty(symbol, usdt_amount / price, price)
                if qty > 0:
                    if place_order(symbol, 'BUY', qty):
                        POSITIONS[symbol] = {'entry': price, 'qty': qty}
                        TOTAL_CAPITAL -= usdt_amount
        time.sleep(60)

if __name__ == '__main__':
    print("🤖 Bắt đầu bot Futures AI...")
    trade_loop()
