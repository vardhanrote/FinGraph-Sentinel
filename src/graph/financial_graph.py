from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx


# ============================================================
# 1. LOAD STOCK PRICE DATA
# ============================================================

def load_stock_data(file_path):
    """
    Load cleaned stock-price data.

    Expected columns:
        date
        Ticker
        Close

    Parameters
    ----------
    file_path : str or Path
        Path to stock_prices_clean.csv

    Returns
    -------
    pandas.DataFrame
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Stock price file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {"date", "Ticker", "Close"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["Ticker", "date"])

    return df


# ============================================================
# 2. CREATE PRICE MATRIX
# ============================================================

def create_price_matrix(df):
    """
    Convert long-format stock data into a price matrix.

    Rows    -> dates
    Columns -> companies
    Values  -> closing prices

    Returns
    -------
    pandas.DataFrame
    """

    price_matrix = df.pivot_table(
        index="date",
        columns="Ticker",
        values="Close",
        aggfunc="last"
    )

    price_matrix = price_matrix.sort_index()

    return price_matrix


# ============================================================
# 3. CALCULATE DAILY RETURNS
# ============================================================

def calculate_returns(price_matrix):
    """
    Calculate percentage returns for each company.

    Returns
    -------
    pandas.DataFrame
    """

    returns = price_matrix.pct_change()

    # Replace infinite values
    returns = returns.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return returns


# ============================================================
# 4. CALCULATE CORRELATION MATRIX
# ============================================================

def calculate_correlation_matrix(returns):
    """
    Calculate Pearson correlation between company returns.

    Returns
    -------
    pandas.DataFrame
    """

    correlation_matrix = returns.corr()

    return correlation_matrix


# ============================================================
# 5. BUILD FINANCIAL GRAPH
# ============================================================

def build_financial_graph(
    correlation_matrix,
    threshold=0.70
):
    """
    Build an undirected weighted financial correlation graph.

    Nodes:
        Companies

    Edges:
        Created when absolute correlation >= threshold

    Edge weight:
        Absolute correlation

    Edge attribute:
        correlation = original signed correlation

    Parameters
    ----------
    correlation_matrix : pandas.DataFrame
        Company correlation matrix.

    threshold : float
        Minimum absolute correlation required to create an edge.

    Returns
    -------
    networkx.Graph
    """

    G = nx.Graph()

    companies = correlation_matrix.columns.tolist()

    # Add every company as a node
    for company in companies:
        G.add_node(company)

    # Add edges
    for i in range(len(companies)):

        for j in range(i + 1, len(companies)):

            company_a = companies[i]
            company_b = companies[j]

            correlation = correlation_matrix.loc[
                company_a,
                company_b
            ]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= threshold:

                G.add_edge(
                    company_a,
                    company_b,
                    weight=abs(correlation),
                    correlation=correlation
                )

    return G


# ============================================================
# 6. CALCULATE GRAPH METRICS
# ============================================================

def calculate_graph_metrics(G):
    """
    Calculate important network metrics.

    Metrics:
        Degree
        Weighted Degree
        Degree Centrality
        Betweenness Centrality
        Eigenvector Centrality
        Community

    Returns
    -------
    pandas.DataFrame
    """

    # --------------------------------------------------------
    # Degree
    # --------------------------------------------------------

    degree = dict(G.degree())

    # --------------------------------------------------------
    # Weighted Degree
    # --------------------------------------------------------

    weighted_degree = dict(
        G.degree(weight="weight")
    )

    # --------------------------------------------------------
    # Degree Centrality
    # --------------------------------------------------------

    degree_centrality = nx.degree_centrality(G)

    # --------------------------------------------------------
    # Betweenness Centrality
    # --------------------------------------------------------

    if G.number_of_edges() > 0:

        betweenness_centrality = nx.betweenness_centrality(
            G,
            weight="weight"
        )

    else:

        betweenness_centrality = {
            node: 0.0
            for node in G.nodes()
        }

    # --------------------------------------------------------
    # Eigenvector Centrality
    # --------------------------------------------------------

    if G.number_of_edges() > 0:

        try:

            eigenvector_centrality = nx.eigenvector_centrality(
                G,
                weight="weight",
                max_iter=1000
            )

        except nx.PowerIterationFailedConvergence:

            eigenvector_centrality = {
                node: 0.0
                for node in G.nodes()
            }

    else:

        eigenvector_centrality = {
            node: 0.0
            for node in G.nodes()
        }

    # --------------------------------------------------------
    # Community Detection
    # --------------------------------------------------------

    if G.number_of_edges() > 0:

        communities = nx.community.greedy_modularity_communities(
            G,
            weight="weight"
        )

        community_mapping = {}

        for community_id, community in enumerate(communities):

            for company in community:

                community_mapping[company] = community_id

    else:

        community_mapping = {
            node: i
            for i, node in enumerate(G.nodes())
        }

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    metrics = []

    for company in G.nodes():

        metrics.append({

            "Ticker": company,

            "Degree": degree.get(
                company,
                0
            ),

            "Weighted_Degree": weighted_degree.get(
                company,
                0.0
            ),

            "Degree_Centrality": degree_centrality.get(
                company,
                0.0
            ),

            "Betweenness_Centrality": betweenness_centrality.get(
                company,
                0.0
            ),

            "Eigenvector_Centrality": eigenvector_centrality.get(
                company,
                0.0
            ),

            "Community": community_mapping.get(
                company,
                -1
            )
        })

    metrics_df = pd.DataFrame(metrics)

    # Sort by weighted degree
    metrics_df = metrics_df.sort_values(
        "Weighted_Degree",
        ascending=False
    )

    metrics_df = metrics_df.reset_index(
        drop=True
    )

    return metrics_df


# ============================================================
# 7. SAVE GRAPH METRICS
# ============================================================

def save_graph_metrics(
    metrics_df,
    output_path
):
    """
    Save graph metrics to CSV.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Graph metrics saved to: {output_path}"
    )


# ============================================================
# 8. PRINT GRAPH SUMMARY
# ============================================================

def print_graph_summary(G, metrics_df):
    """
    Print a summary of the financial graph.
    """

    print("\n" + "=" * 60)
    print("FINANCIAL GRAPH SUMMARY")
    print("=" * 60)

    print(
        f"Graph nodes: {G.number_of_nodes()}"
    )

    print(
        f"Graph edges: {G.number_of_edges()}"
    )

    if "Community" in metrics_df.columns:

        number_of_communities = (
            metrics_df["Community"]
            .nunique()
        )

    else:

        number_of_communities = 0

    print(
        f"Number of communities: "
        f"{number_of_communities}"
    )

    print("\nTop companies:")

    columns_to_show = [
        "Ticker",
        "Degree",
        "Weighted_Degree",
        "Degree_Centrality",
        "Betweenness_Centrality",
        "Eigenvector_Centrality",
        "Community"
    ]

    print(
        metrics_df[
            columns_to_show
        ].head(10).to_string(index=False)
    )

    print("=" * 60)


# ============================================================
# 9. COMPLETE PIPELINE
# ============================================================

def build_financial_graph_pipeline(
    input_path,
    output_path,
    threshold=0.70
):
    """
    Complete financial graph pipeline.

    Steps:
        1. Load stock data
        2. Create price matrix
        3. Calculate returns
        4. Calculate correlations
        5. Build graph
        6. Calculate graph metrics
        7. Save metrics

    Returns
    -------
    G : networkx.Graph
    metrics_df : pandas.DataFrame
    correlation_matrix : pandas.DataFrame
    """

    # --------------------------------------------------------
    # Step 1: Load data
    # --------------------------------------------------------

    print("Step 1: Loading stock data...")

    df = load_stock_data(
        input_path
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Step 2: Price matrix
    # --------------------------------------------------------

    print(
        "Step 2: Creating price matrix..."
    )

    price_matrix = create_price_matrix(
        df
    )

    print(
        f"Price matrix shape: "
        f"{price_matrix.shape}"
    )

    # --------------------------------------------------------
    # Step 3: Returns
    # --------------------------------------------------------

    print(
        "Step 3: Calculating returns..."
    )

    returns = calculate_returns(
        price_matrix
    )

    # --------------------------------------------------------
    # Step 4: Correlations
    # --------------------------------------------------------

    print(
        "Step 4: Calculating correlations..."
    )

    correlation_matrix = calculate_correlation_matrix(
        returns
    )

    # --------------------------------------------------------
    # Step 5: Graph
    # --------------------------------------------------------

    print(
        "Step 5: Building financial graph..."
    )

    G = build_financial_graph(
        correlation_matrix,
        threshold=threshold
    )

    # --------------------------------------------------------
    # Step 6: Metrics
    # --------------------------------------------------------

    print(
        "Step 6: Calculating graph metrics..."
    )

    metrics_df = calculate_graph_metrics(
        G
    )

    # --------------------------------------------------------
    # Step 7: Save
    # --------------------------------------------------------

    print(
        "Step 7: Saving graph metrics..."
    )

    save_graph_metrics(
        metrics_df,
        output_path
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_graph_summary(
        G,
        metrics_df
    )

    return (
        G,
        metrics_df,
        correlation_matrix
    )


# ============================================================
# 10. RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    INPUT_PATH = Path(
        "../data/processed/stock_prices_clean.csv"
    )

    OUTPUT_PATH = Path(
        "../data/processed/graph_metrics.csv"
    )

    GRAPH_THRESHOLD = 0.70

    G, final_graph_metrics, correlation_matrix = (
        build_financial_graph_pipeline(
            input_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
            threshold=GRAPH_THRESHOLD
        )
    )

    print("\nFinancial graph pipeline completed successfully.")
