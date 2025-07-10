from binance.client import Client
from binance.enums import *
import pandas as pd
import ta
import time
import os

API_KEY = os.environ.get("BINANCE_API_KEY")
API_SECRET = os.environ.get("BINANCE_API_SECRET")
client = Client(API_KEY, API_SECRET)

symbol = 'BTCUSDT'
quantity = 0.001

def get_klines(symbol, interval='1h', limit=100):
    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(candles, columns=[
        'time','open','high','low','close','volume',
        'close_time','qav','trades','tbbav','tbqav','ignore'])
    df['close'] = df['close'].astype(float)
    return df

def get_signal(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['ema20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['ema50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()

    if df['rsi'].iloc[-1] < 30 and df['ema20'].iloc[-1] > df['ema50'].iloc[-1]:
        return 'BUY'
    elif df['rsi'].iloc[-1] > 70 and df['ema20'].iloc[-1] < df['ema50'].iloc[-1]:
        return 'SELL'
    return 'HOLD'

def place_order(side):
    print(f"Placing {side} order for {symbol}")
    # Lệnh demo: không giao dịch thật
    # Muốn thật thì bỏ comment bên dưới
    # client.create_order(symbol=symbol, side=SIDE_BUY if side=='BUY' else SIDE_SELL, type=ORDER_TYPE_MARKET, quantity=quantity)

while True:
    try:
        df = get_klines(symbol)
        signal = get_signal(df)
        print(f"Signal: {signal}")
        if signal in ['BUY', 'SELL']:
            place_order(signal)
        time.sleep(300)
    except Exception as e:
        print(f"Lỗi: {e}")
        time.sleep(60)
