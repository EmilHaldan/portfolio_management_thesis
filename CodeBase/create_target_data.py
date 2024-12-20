import pandas as pd
import os
import sys
import win32cred
import requests
from datetime import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from eodhd import APIClient
from ticker_loader import load_SPY_components
import sqlite3
from tqdm import tqdm
import time 
import pickle
from SQLite_tools import query_stock_data, query_and_combine_TA_data, create_and_upload_table 
from SQLite_tools import check_if_close_price_exists, fetch_all_trading_days, get_all_quarters, fetch_all_tickers


import pandas as pd
import numpy as np

def create_target_labels(future_days=5):
    """
    Assigns classification targets to each stock based on percentage change over a given future_days period.

    Args:
        future_days (int): Number of steps ahead to calculate the percentage change (default: 5 , corresponding to 1 week ahead).

    Returns:
        pd.DataFrame: DataFrame with an added 'target' column indicating the classification target.
    """
    # Sort the data to ensure proper calculation
    
    all_stock  = fetch_all_tickers()
    all_trading_days = fetch_all_trading_days(zero_pad=True)

    # Calculate percentage change for each stock
    target_data_sets = []
    unmodified_data_sets = []
    for stock in tqdm(all_stock, desc="Calculating target labels for all stocks being considered."):
        price_data = query_stock_data(table="Price_Data", ticker=stock)[["Date", "Close"]]
        price_data['pct_change'] = price_data['Close'].pct_change(
                                periods=future_days)
        price_data["pct_change"] = price_data["pct_change"].shift(-5)
        
        price_data.drop(columns=['Close'], inplace=True)
        unmodified_data_sets.append(price_data.copy(deep=True))
        date_data = pd.DataFrame(all_trading_days, columns=['Date'])
        merged_data = date_data.merge(price_data, how='left', on='Date')
        merged_data['Ticker'] = stock
        merged_data['pct_change'] = merged_data['pct_change'].fillna(0)
        target_data_sets.append(merged_data.copy(deep=True))

    target_data = pd.concat(target_data_sets, axis=0)
    target_data.reset_index(drop=True, inplace=True)
    target_data = target_data[['Date', 'Ticker', 'pct_change']]

    unmodified_target_data = pd.concat(unmodified_data_sets, axis=0)

    mean_change = unmodified_target_data['pct_change'].mean(skipna=True)
    std_change = unmodified_target_data['pct_change'].std(skipna=True)

    bins = [-np.inf, 
            mean_change - std_change * 0.7, 
            mean_change - std_change * 0.20,  
            mean_change + std_change * 0.20, 
            mean_change + std_change * 0.7, 
            np.inf]
    
    labels = [-2, -1, 0, 1, 2]

    target_data['target'] = pd.cut(target_data['pct_change'], bins=bins, labels=labels)
    unmodified_target_data['target'] = pd.cut(unmodified_target_data['pct_change'], bins=bins, labels=labels)
    target_data.rename(columns={'pct_change':'Pct_Change'}, inplace=True)

    print("Target Data labels distribution")
    print(target_data['target'].value_counts())
    print("")
    print("Unmodified Target Data labels distribution")
    print(unmodified_target_data['target'].value_counts())
    print("")

    create_and_upload_table( target_data, "Target_Data", 
                            column_names = ["Date", "Ticker", "Pct_Change", 'target'] , 
                            primary_key = ["Date", "Ticker"])

    # target_data.to_excel('../Data/Aux_datasets_while_dev/Target_data.xlsx', index=False)

    return target_data


# Example usage
if __name__ == "__main__":


    result = create_target_labels()
    print(result)

