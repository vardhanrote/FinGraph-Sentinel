import os
import yfinance as yf
import pandas as pd

from companies import COMPANIES


RAW_DATA_PATH = "data/raw/stock_prices"


def collect_stock_data():

    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    for ticker, company_name in COMPANIES.items():

        print(f"Downloading data for {company_name} ({ticker})...")

        stock = yf.Ticker(ticker)

        data = stock.history(
            period="5y",
            interval="1d"
        )

        if data.empty:
            print(f"No data found for {ticker}")
            continue

        data = data.reset_index()

        data["ticker"] = ticker
        data["company"] = company_name

        file_path = os.path.join(
            RAW_DATA_PATH,
            f"{ticker}.csv"
        )

        data.to_csv(file_path, index=False)

        print(f"Saved: {file_path}")


if __name__ == "__main__":
    collect_stock_data()