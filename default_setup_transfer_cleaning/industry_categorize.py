import yfinance as yf
import pandas as pd
import os

# Directory containing the CSV files
csv_path = 'stock/stock_data'  # Replace with your actual path
files = [f for f in os.listdir(csv_path) if f.endswith("_historical_data.csv")]

# Function to get industry information using yfinance
def get_industry_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return info.get('industry', 'Unknown')
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return 'Unknown'

# Extract stock tickers from file names
stock_tickers = [os.path.splitext(f)[0].replace('_historical_data', '') for f in files]

# Dictionary to hold the ticker and corresponding industry
industry_info = {}

for ticker in stock_tickers:
    industry = get_industry_info(ticker)
    industry_info[ticker] = industry
    print(f"{ticker}: {industry}")

# Save the industry information to a CSV file
industry_info_df = pd.DataFrame(list(industry_info.items()), columns=['Ticker', 'Industry'])
output_file = 'industry_category/stock_industry_info.csv'
industry_info_df.to_csv(output_file, index=False)

print(f"Industry information saved to {output_file}")
