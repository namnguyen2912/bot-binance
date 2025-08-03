import os
import requests
import zipfile
import pandas as pd

# Thư mục lưu dữ liệu
DATA_DIR = "/content/binance_data"
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
symbol = "BTCUSDT"
interval = "1m"

# Danh sách năm-tháng từ 2017 đến 2025
years = range(2017, 2025)
months = range(1, 13)

def download_month(symbol, interval, year, month):
    """Tải 1 file ZIP của tháng"""
    month_str = f"{month:02d}"
    file_name = f"{symbol}-{interval}-{year}-{month_str}.zip"
    url = f"{BASE_URL}/{symbol}/{interval}/{file_name}"
    save_path = os.path.join(DATA_DIR, file_name)

    if os.path.exists(save_path.replace(".zip", ".csv")):
        print(f"✅ Đã có dữ liệu: {file_name}")
        return save_path

    print(f"⬇️ Đang tải: {url}")
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(save_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(save_path)  # Xóa file zip
        print(f"✅ Đã tải xong {file_name}")
        return save_path
    else:
        print(f"❌ Không có dữ liệu: {file_name}")
        return None

# === Tải dữ liệu tất cả tháng ===
for y in years:
    for m in months:
        download_month(symbol, interval, y, m)

# === Ghép tất cả file CSV ===
print("🔗 Đang ghép dữ liệu...")
csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
csv_files.sort()

all_data = []
for file in csv_files:
    df = pd.read_csv(file, header=None)
    df.columns = ["open_time","open","high","low","close","volume",
                  "close_time","quote_asset_volume","number_of_trades",
                  "taker_buy_base","taker_buy_quote","ignore"]
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    all_data.append(df[["open_time","open","high","low","close","volume"]])

final_df = pd.concat(all_data, ignore_index=True)
final_path = os.path.join(DATA_DIR, f"{symbol}_{interval}_history.csv")
final_df.to_csv(final_path, index=False)
print(f"🎉 Hoàn tất! File cuối cùng: {final_path}")
