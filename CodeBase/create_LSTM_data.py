import pandas as pd
import os
import numpy as np
from datetime import datetime
from ticker_loader import load_SPY_components
import sqlite3
from tqdm import tqdm
import time 
import pickle
import torch
from torch_geometric.data import Data
from monthly_backtester import load_index_to_ticker_dict

from SQLite_tools import query_stock_data, query_and_combine_TA_data, create_and_upload_table 
from SQLite_tools import check_if_close_price_exists, fetch_all_trading_days, get_all_quarters, fetch_all_tickers

import numpy as np
from tqdm import tqdm


def create_windowed_data(window_size):
    """
    Creates windowed sequences for LSTM from a time-series dataset, storing each window directly in the dictionary.

    Args:
        window_size (int): The number of time steps in each window.

    Returns:
        list: A list of dictionaries, each containing a window of data and its metadata.
    """
    windowed_data = []
    instance_id = 0
    index_to_ticker, ticker_to_index = load_index_to_ticker_dict()

    all_stock_ticker = fetch_all_tickers()

    for ticker in tqdm(all_stock_ticker, desc="Processing windowed data for each ticker"):

        active_dates = query_stock_data(table="Active_Stocks", ticker=ticker)
        active_dates = active_dates[active_dates["Active"] == 1]
        start_date = active_dates["Date"].values[0]
        end_date = active_dates["Date"].values[-1]
        stock_data = query_stock_data(table="ML_Feature_Table", ticker=ticker, start_date=start_date, end_date=end_date)
        total_target_data = query_stock_data(table="Target_Data", ticker=ticker, start_date=start_date, end_date=end_date)[["Date", "target"]]

        # Merge targets and preprocess
        stock_data = stock_data.merge(total_target_data, on="Date", how="left")
        stock_data["target"] = stock_data["target"].fillna(0).astype(np.int32)
        total_target_data = stock_data["target"].values
        dates = stock_data['Date'].values
        stock_data.drop(columns=["Ticker","Active","target", "Date"], inplace=True)

        # Skip tickers with insufficient data
        if len(stock_data) < window_size:
            continue

        stock_data_np = stock_data.to_numpy()

        for i in range(len(stock_data_np) - window_size + 1):
            # Extract window and metadata
            start_idx = i
            end_idx = i + window_size
            window = stock_data_np[start_idx:end_idx]
            target = total_target_data[end_idx - 1]
            window_dates = dates[start_idx:end_idx]

            # Create dictionary for the instance
            instance = {
                'instance_id': instance_id,
                'ticker': ticker,
                'ticker_idx': ticker_to_index[ticker],
                'window': window,  # Store the actual window data
                'target': target,
                'start_date': window_dates[0],
                'end_date': window_dates[-1],
                'prediction_date': window_dates[-1],
                'window_dates': window_dates
            }
            windowed_data.append(instance)
            instance_id += 1

    return windowed_data


def save_windowed_data(windowed_data, output_dir, window_size):
    """
    Saves windowed data into a pickle file.

    Args:
        windowed_data (list): List of windowed data dictionaries.
        output_dir (str): Directory to save the data.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"lstm_windowed_data_{window_size}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(windowed_data, f)
    print(f"Windowed data saved to {output_path}")


if __name__ == "__main__":
    # Parameters
    WINDOW_SIZE = 5
    OUTPUT_DIR = "../Data/LSTM_Data/".format(WINDOW_SIZE)

    # Create windowed data
    windowed_data = create_windowed_data(window_size=WINDOW_SIZE)

    # Save the windowed data
    save_windowed_data(windowed_data, OUTPUT_DIR, WINDOW_SIZE) 
