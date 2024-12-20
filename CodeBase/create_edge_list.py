import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from ticker_loader import load_SPY_components
import sqlite3
from tqdm import tqdm
import time 
import pickle
import random as rn
from datetime import datetime
from datetime import timedelta
from statsmodels.tsa.stattools import coint
import holidays

from create_financial_database import get_credentials, populate_database
from SQLite_tools import query_stock_data, check_if_close_price_exists
from SQLite_tools import fetch_all_trading_days, fetch_all_tickers


import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import numpy as np
from SQLite_tools import query_stock_data, fetch_all_trading_days

def load_edge_dict(dict_type: str):
    """
    Load an existing edge dictionary if available, otherwise return an empty dictionary.
    """
    pickle_file = f'../Data/Edges/{dict_type}.pkl'
    if os.path.exists(pickle_file):
        with open(pickle_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_edge_dict(dict_type: str, correlation_dict: dict):
    """
    Save the correlation dictionary as a pickle file.
    """
    pickle_file = f'../Data/Edges/{dict_type}.pkl'
    with open(pickle_file, 'wb') as f:
        pickle.dump(correlation_dict, f)


def compute_correlations(date, price_df, range_for_correlation, threshold, min_good_edges_to_add):
    """
    Compute correlations for a given date.
    """
    end_idx = price_df.index.get_loc(date)
    start_idx = max(end_idx - range_for_correlation, 0)
    window_data = price_df.iloc[start_idx:end_idx]

    # Calculate pairwise correlations
    correlation_matrix = window_data.corr()
    correlation_matrix.fillna(0, inplace=True)

    stocks_node_degree = {stock: 0 for stock in correlation_matrix.columns}
    correlations = {}

    edge_list = []
    for stock_1 in correlation_matrix.columns:
        for stock_2 in correlation_matrix.columns:
            if stock_1 != stock_2:
                correlation = correlation_matrix.loc[stock_1, stock_2]
                if abs(correlation) >= threshold:
                    edge = tuple(sorted((stock_1, stock_2)))
                    edge_list.append((edge, abs(correlation)))

    sorted_edges = sorted(edge_list, key=lambda x: x[1], reverse=True)

    for edge, correlation in sorted_edges:
        stock_1, stock_2 = edge
        if stocks_node_degree[stock_1] < min_good_edges_to_add and stocks_node_degree[stock_2] < min_good_edges_to_add:
            if edge not in correlations:
                scaled_correlation = (correlation + 1) / 2
                correlations[edge] = {"correlation": scaled_correlation}
                stocks_node_degree[stock_1] += 1
                stocks_node_degree[stock_2] += 1

    return date, correlations
    

def calculate_matrixwise_correlations(dict_type: str, range_for_correlation: int = 43):
    """
    Sequential calculation of pairwise correlations between stocks.
    """
    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    viable_dates = fetch_all_trading_days(zero_pad=False)
    stocks = query_stock_data(table="Stocks")["Ticker"].to_list()
    active_stocks = query_stock_data(table="Active_Stocks", ticker="all_stocks")

    # Query data for all stocks and create a DataFrame
    all_price_data = {}
    for ticker in stocks:
        price_data = query_stock_data(table="Price_Data", ticker=ticker)
        if price_data.empty:
            continue
        price_data['Date'] = price_data['Date'].apply(lambda x: x[:10])
        all_price_data[ticker] = price_data.set_index("Date")["Close"]

    price_df = pd.DataFrame(all_price_data).sort_index()
    conn.close()

    # Load existing correlation dictionary or initialize an empty one
    correlations_dict = load_edge_dict(dict_type)
    threshold = 0.5
    max_good_edges_to_add = 5

    # Process each date sequentially
    for idx,date in tqdm(enumerate(viable_dates), desc=f"Processing Correlations", total=len(viable_dates)):
        if idx < range_for_correlation or date in correlations_dict:
            # Skip already processed dates  AND  Skip the first few dates of price data
            continue

        # Compute correlations for the current date
        _, correlations = compute_correlations(
            date, price_df, range_for_correlation, threshold, max_good_edges_to_add)
        correlations_dict[date] = correlations

        # Save progress after every 100 dates
        if len(correlations_dict) % 100 == 0:
            save_edge_dict(dict_type, correlations_dict)

    # Save final results
    save_edge_dict(dict_type, correlations_dict)
    return correlations_dict



if __name__ == "__main__":
    if not os.path.exists("../Data/Edges"):
        os.mkdir("../Data/Edges")

    calculate_matrixwise_correlations(dict_type="correlation_short_term_test", range_for_correlation=43)



