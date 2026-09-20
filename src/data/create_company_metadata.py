import pandas as pd

from companies import COMPANIES


def create_metadata():

    records = []

    for ticker, company in COMPANIES.items():

        records.append({
            "ticker": ticker,
            "company": company
        })

    df = pd.DataFrame(records)

    df.to_csv(
        "data/raw/company_metadata.csv",
        index=False
    )

    print("Company metadata created successfully.")


if __name__ == "__main__":
    create_metadata()