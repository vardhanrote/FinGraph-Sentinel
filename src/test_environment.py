import sys
import pandas as pd
import numpy as np
import networkx as nx
import yfinance as yf


def main():
    print("FinGraph-Sentinel Environment Test")
    print("----------------------------------")
    print(f"Python version: {sys.version}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    print(f"NetworkX version: {nx.__version__}")

    print("\nEnvironment setup successful!")


if __name__ == "__main__":
    main()