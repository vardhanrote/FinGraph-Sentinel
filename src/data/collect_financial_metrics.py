import os
import pandas as pd
import yfinance as yf

from companies import COMPANIES


OUTPUT_PATH = "data/raw/financial_metrics.csv"


def collect_financial_metrics():

    records = []

    for ticker, company_name in COMPANIES.items():

        print(f"Collecting financial metrics for {company_name}...")

        stock = yf.Ticker(ticker)

        info = stock.info

        record = {
            "ticker": ticker,
            "company": company_name,
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "profit_margin": info.get("profitMargins"),
            "revenue_growth": info.get("revenueGrowth"),
            "return_on_equity": info.get("returnOnEquity"),
            "debt_to_equity": info.get("debtToEquity"),
        }

        records.append(record)

    df = pd.DataFrame(records)

    os.makedirs("data/raw", exist_ok=True)

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(f"\nSaved financial metrics to {OUTPUT_PATH}")


if __name__ == "__main__":
    collect_financial_metrics()