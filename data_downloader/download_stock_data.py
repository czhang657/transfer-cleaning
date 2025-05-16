import os
import pandas as pd
import yfinance as yf
import requests
from time import sleep

# === Configuration ===
output_dir = "stock/stock_data"
os.makedirs(output_dir, exist_ok=True)

# Set how many tickers to download (e.g., 100 for testing). Set to None to download all.
MAX_TICKERS = 10  # Change to None for all tickers

# === Fetch NASDAQ ticker list from NASDAQ Trader ===
def get_nasdaq_tickers():
    url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/nasdaqlisted.txt"
    try:
        df = pd.read_csv(url, sep="|")
        tickers = df["Symbol"]
        tickers = tickers[pd.to_numeric(tickers, errors="coerce").isna()]  # 排除数值行
        tickers = tickers.dropna().astype(str)
        tickers = [t for t in tickers if t.isalpha()]
        return tickers
    except Exception as e:
        print(f"Failed to download NASDAQ tickers: {e}")
        return []

        return tickers
    except Exception as e:
        print(f"Failed to download NASDAQ tickers: {e}")
        return []

# === Download function for each stock ===
def download_stock(ticker):
    print(f"📥 Downloading {ticker}...")
    try:
        df = yf.download(ticker, period="max", auto_adjust=False)
        if df.empty:
            print(f"{ticker}: No data found (possibly delisted).")
            return

        df.reset_index(inplace=True)

        # Flatten multi-level columns caused by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.to_flat_index()

        # Set clean column names manually
        df.columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]

        output_path = os.path.join(output_dir, f"{ticker}_historical_data.csv")
        df.to_csv(output_path, index=False)
        print(f"{ticker}: {len(df)} rows saved.")

    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
    sleep(1)


# === Main process ===
if __name__ == "__main__":
    tickers = get_nasdaq_tickers()
    if MAX_TICKERS:
        tickers = tickers[:MAX_TICKERS]

    print(f"Preparing to download {len(tickers)} NASDAQ tickers...\n")

    for ticker in tickers:
        download_stock(ticker)

    print("\n Download completed.")
