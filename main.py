# === [Binance Futures AI Trading Bot - Multi Coin, Improved Algo] ===

import time
import numpy as np
import pandas as pd
from binance.um_futures import UMFutures
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

api_key = "YOUR_API_KEY"
api_secret = "YOUR_API_SECRET"

client = UMFutures(key=api_key, secret=api_secret)

# === CONFIG ===
CAPITAL = 1000.0  # virtual capital in USDT
POSITION_SIZE = 0.05  # max 5% per trade
TAKE_PROFIT = 0.03  # +3%
STOP_LOSS = -0.015  # -1.5%
TOP_N_SYMBOLS = 3  # number of coins to trade
INTERVAL = "15m"
CANDLE_LIMIT = 150

positions = {}  # virtual positions

# === Feature Engineering ===
def calculate_features(df):
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(10).std()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['ema_fast'] = df['close'].ewm(span=5).mean()
    df['ema_slow'] = df['close'].ewm(span=20).mean()
    df['macd'] = df['ema_fast'] - df['ema_slow']
    df['bb_width'] = (df['close'].rolling(20).max() - df['close'].rolling(20).min()) / df['close']
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# === Get top trading pairs by volume ===
def get_top_symbols(limit=TOP_N_SYMBOLS):
    tickers = client.ticker_24hr()
    usdt_pairs = [t for t in tickers if t['symbol'].endswith('USDT') and not t['symbol'].endswith('BUSD')]
    sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
    return [s['symbol'] for s in sorted_pairs[:limit]]

# === Train ML model ===
def train_model(df):
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    df['target'] = np.where(df['future_return'] > 0.007, 1, np.where(df['future_return'] < -0.007, -1, 0))
    df.dropna(inplace=True)
    features = ['return', 'volatility', 'rsi', 'macd', 'bb_width']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[features])
    y = df['target']
    clf = GradientBoostingClassifier()
    clf.fit(X, y)
    return clf, scaler, features

# === Execute Trade Logic ===
def run_bot(symbol):
    global CAPITAL
    candles = client.klines(symbol=symbol, interval=INTERVAL, limit=CANDLE_LIMIT)
    df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume","close_time","qav","trades","taker_base","taker_quote","ignore"])
    df['close'] = df['close'].astype(float)
    df = calculate_features(df)

    model, scaler, feats = train_model(df.copy())
    latest = df.iloc[-1:]
    X_live = scaler.transform(latest[feats])
    prediction = model.predict(X_live)[0]
    proba = model.predict_proba(X_live)[0][prediction + 1]

    print(f"🔍 {symbol} | Predict: {prediction} | Proba: {proba:.2f} | Capital: {CAPITAL:.2f}")

    # Check if holding
    if symbol in positions:
        pos = positions[symbol]
        current_price = float(client.ticker_price(symbol)['price'])
        pnl = (current_price - pos['entry_price']) / pos['entry_price']
        if pnl >= TAKE_PROFIT or pnl <= STOP_LOSS:
            usdt_gained = pos['amount'] * current_price
            CAPITAL += usdt_gained
            print(f"✅ Exit {symbol} | PnL: {pnl:.2%} | Capital now: {CAPITAL:.2f}")
            del positions[symbol]
        return

    # Open new position if conditions met
    if prediction == 1 and proba > 0.7 and symbol not in positions:
        size_usdt = CAPITAL * POSITION_SIZE
        price = float(client.ticker_price(symbol)['price'])
        quantity = round(size_usdt / price, 6)
        positions[symbol] = {
            "entry_price": price,
            "amount": quantity,
        }
        CAPITAL -= size_usdt
        print(f"📈 BUY {symbol} | Qty: {quantity} | Price: {price:.2f} | Capital left: {CAPITAL:.2f}")

# === Main Loop ===
if __name__ == "__main__":
    while True:
        top_symbols = get_top_symbols()
        for sym in top_symbols:
            try:
                run_bot(sym)
            except Exception as e:
                print(f"❌ Error with {sym}: {e}")
        time.sleep(900)  # every 15 min
