import os
import time
import numpy as np
import pandas as pd
from binance.um_futures import UMFutures
from binance.error import ClientError
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

# =========================
# ENV & CLIENT (Futures Testnet)
# =========================
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

if not API_KEY or not API_SECRET:
    raise ValueError("❌ Vui lòng set ENV API_KEY và API_SECRET (Render: Settings ➜ Environment).")

# Testnet Futures base_url:
client = UMFutures(key=API_KEY, secret=API_SECRET, base_url="https://testnet.binancefuture.com")

# =========================
# CONFIG
# =========================
INTERVAL = "15m"
TOTAL_CAPITAL = 1000.0           # vốn ảo để quản lý
TRADE_PERCENTAGE = 0.05          # 5% mỗi lệnh
TP_PCT = 0.03                    # chốt lời +3%
SL_PCT = 0.015                   # cắt lỗ -1.5%

POSITIONS = {}                   # {symbol: {"entry": float, "qty": float}}
symbol_info_cache = {}           # cache exchangeInfo per symbol


# =========================
# EXCHANGE INFO HELPERS
# =========================
def get_symbol_info(symbol: str):
    """Cache exchangeInfo cho từng symbol."""
    if symbol in symbol_info_cache:
        return symbol_info_cache[symbol]
    try:
        info = client.exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                symbol_info_cache[symbol] = s
                return s
    except Exception as e:
        print(f"⚠️ Lỗi lấy exchange_info cho {symbol}: {e}")
    return None


def adjust_qty(symbol: str, raw_qty: float, price: float) -> float:
    """Làm tròn qty theo LOT_SIZE và đảm bảo tối thiểu theo MIN_NOTIONAL."""
    info = get_symbol_info(symbol)
    if not info:
        return max(0.0, round(raw_qty, 3))

    step_size = 0.0
    min_qty = 0.0
    min_notional = 0.0

    for f in info["filters"]:
        ftype = f.get("filterType")
        if ftype == "LOT_SIZE":
            step_size = float(f["stepSize"])
            min_qty = float(f["minQty"])
        elif ftype == "MIN_NOTIONAL":
            # UM Futures filter
            min_notional = float(f["notional"])

    # đảm bảo >= minQty
    qty = max(raw_qty, min_qty)

    # đảm bảo notional tối thiểu
    if price * qty < min_notional:
        qty = min_notional / price

    # làm tròn theo step_size
    if step_size > 0:
        precision = int(round(-np.log10(step_size)))
        qty = round(qty - (qty % step_size), precision)
    else:
        qty = round(qty, 3)

    return max(0.0, qty)


# =========================
# DATA & FEATURES
# =========================
def fetch_klines(symbol: str, interval: str = INTERVAL, limit: int = 300):
    try:
        klines = client.klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(
            klines,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
            ],
        )
        df = df.astype({"close": float, "volume": float, "high": float, "low": float})
        return df
    except ClientError as e:
        print(f"❌ Lỗi fetch klines {symbol}: {e.error_message}")
        return None


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["return"] = df["close"].pct_change().fillna(0)
    df["ema10"] = df["close"].ewm(span=10).mean()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["rsi"] = compute_rsi(df["close"])
    # label: tăng trong 3 nến tới > 0.2%
    df["future_return"] = df["close"].shift(-3) / df["close"] - 1
    df["target"] = (df["future_return"] > 0.002).astype(int)
    return df.dropna()


# =========================
# MODEL
# =========================
def train_model(df_feat: pd.DataFrame):
    features = ["return", "ema10", "ema20", "rsi"]
    X = df_feat[features]
    y = df_feat["target"]

    # fallback nếu dữ liệu kém
    if len(X) < 120 or y.nunique() < 2:
        return None

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = LGBMClassifier(n_estimators=120, max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train)
    return model


def get_prediction(model, df_feat: pd.DataFrame) -> int:
    if model is None:
        # fallback heuristic: EMA cross + RSI filter
        last = df_feat.iloc[-1]
        signal = 1 if (last["ema10"] > last["ema20"] and last["rsi"] > 55) else 0
        return signal
    latest = df_feat[["return", "ema10", "ema20", "rsi"]].iloc[-1:]
    return int(model.predict(latest)[0])


# =========================
# SYMBOL SELECTION
# =========================
def get_top_symbols(limit: int = 5):
    """Lấy top USDT symbols theo quoteVolume (24h) trên Futures."""
    try:
        tickers = client.ticker_24hr_price_change()
        df = pd.DataFrame(tickers)
        df["quoteVolume"] = pd.to_numeric(df["quoteVolume"], errors="coerce")
        df = df[df["symbol"].str.endswith("USDT") & ~df["symbol"].str.contains("1000")]
        top = (
            df.sort_values("quoteVolume", ascending=False)["symbol"]
            .head(limit)
            .tolist()
        )
        return top if top else ["BTCUSDT"]
    except ClientError as e:
        print(f"⚠️ Lỗi lấy top symbols: {e.error_message}")
        return ["BTCUSDT"]


# =========================
# ORDER
# =========================
def place_order(symbol: str, side: str, qty: float):
    try:
        resp = client.new_order(symbol=symbol, side=side, type="MARKET", quantity=qty)
        print(f"✅ Lệnh {side} {symbol} OK | qty={qty} | orderId={resp.get('orderId')}")
        return True
    except ClientError as e:
        print(f"❌ Lỗi lệnh {side} {symbol}: {e.error_message}")
        return False


# =========================
# POSITION MGMT (TP/SL)
# =========================
def check_positions():
    global POSITIONS, TOTAL_CAPITAL
    for symbol, pos in list(POSITIONS.items()):
        try:
            price = float(client.ticker_price(symbol)["price"])
        except ClientError as e:
            print(f"⚠️ Lỗi lấy giá {symbol}: {e.error_message}")
            continue

        pnl = (price - pos["entry"]) / pos["entry"]
        if pnl >= TP_PCT or pnl <= -SL_PCT:
            side = "SELL"  # đóng long đơn giản
            print(f"📤 Đóng {symbol} | Entry: {pos['entry']:.4f} | Now: {price:.4f} | PnL: {pnl*100:.2f}%")
            if place_order(symbol, side, pos["qty"]):
                # cộng lại vốn ảo theo notional hiện tại
                TOTAL_CAPITAL += pos["qty"] * price
                del POSITIONS[symbol]


# =========================
# MAIN LOOP
# =========================
def trade_loop():
    global TOTAL_CAPITAL
    while True:
        check_positions()

        symbols = get_top_symbols(limit=5)
        for symbol in symbols:
            df = fetch_klines(symbol, INTERVAL, limit=300)
            if df is None or len(df) < 100:
                continue

            df_feat = add_features(df)
            if df_feat.empty:
                continue

            model = train_model(df_feat)
            pred = get_prediction(model, df_feat)
            price = float(df_feat["close"].iloc[-1])

            print(f"\n📈 {symbol} | Giá: {price:.2f} | AI: {pred}")

            # chỉ mua nếu không có vị thế
            if pred == 1 and symbol not in POSITIONS and TOTAL_CAPITAL > 10:
                usdt_amount = TOTAL_CAPITAL * TRADE_PERCENTAGE
                raw_qty = usdt_amount / price
                qty = adjust_qty(symbol, raw_qty, price)
                if qty <= 0:
                    print(f"⚠️ Qty sau khi làm tròn = 0 (notional không đủ) → bỏ qua {symbol}")
                    continue

                if place_order(symbol, "BUY", qty):
                    POSITIONS[symbol] = {"entry": price, "qty": qty}
                    TOTAL_CAPITAL -= usdt_amount
            else:
                print("⏸️ Không hành động.")

        time.sleep(60)


if __name__ == "__main__":
    print("🤖 Bắt đầu bot Futures AI (Testnet)…")
    trade_loop()
