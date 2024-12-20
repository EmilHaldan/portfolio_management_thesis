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
from SQLite_tools import check_if_close_price_exists, fetch_all_trading_days, get_all_quarters



def create_and_combine_node_feature_data(ticker : str):
    earnings_data = query_stock_data(table="Normalized_Earnings_Data", ticker=ticker)
    fundamental_data = query_stock_data(table="Normalized_Fundamental_Data", ticker=ticker)
    price_data = query_stock_data(table="Normalized_Price_Data", ticker=ticker)
    ta_data = query_stock_data(table="Normalized_TA_Data", ticker=ticker)
    valuation_data = query_stock_data(table="Normalized_Valuation_Data", ticker=ticker)
    
    data_dict = {}
    data_dict["Price"] = price_data
    data_dict["Earnings"] = earnings_data
    data_dict["Fundamental"] = fundamental_data
    data_dict["TA"] = ta_data
    data_dict["Valuation"] = valuation_data

    for key,df in data_dict.items():
        if df.empty:
            return data_dict, False

    return data_dict, True


def re_index(df, keep_ticker=False):
    """
    Re-indexes the DataFrame by date.
    """
    # Convert the "Date" column to string 
    df["Date"] = df["Date"].astype(str)
    if len(df.loc[0,"Date"]) > 10: 
        df["Date"] = df["Date"].apply(lambda x: x[:10])

    # Set the "Date" column as the index
    df = df.set_index("Date")
    if not keep_ticker:
        for col in ["Ticker", "EffectiveDate"]:
            if col in df.columns:
                df = df.drop(columns=[col])

    return df


def create_date_spine(all_trading_days):
    """
    Creates a DataFrame with all trading days as the index.
    """
    date_spine = pd.DataFrame(columns = ["Date"], data = all_trading_days)
    date_spine = re_index(date_spine, keep_ticker=False)
    return date_spine


def remove_data_if_ticker_inactive(df, ticker):
    """
    Sets the Data in the dataframe to 0 if the ticker is inactive.
    """

    ticker_active = query_stock_data(table="Active_Stocks", ticker=ticker)
    ticker_active.drop(columns=["Ticker"], inplace=True)
    df = df.merge(ticker_active, on="Date", how="left")

    columns_to_zero = [col for col in df.columns if col not in ["Ticker", "Date", "Active"]]
    df.loc[df["Active"] == 0, columns_to_zero] = 0

    df.loc[df["Active"] == 0, "Active"] = -1 # Indicator that the stock is inactive and yields no returns in the target.

    return df


def add_sector_data(df, ticker):
    sector_data = query_stock_data(table="Sector_1hot_encoded", ticker=ticker)
    sector_data.drop(columns=["Ticker"], inplace=True)
    for col in sector_data.columns:
        df[col] = sector_data[col].values[0]
    return df


def align_and_merge_feature_data(datasets, all_trading_days, ticker):
    """
    Aligns multiple datasets by date and merges them into a single DataFrame.
    """
    dfs = [create_date_spine(all_trading_days)]
    for key,df in datasets.items():
        if key == "Price":
            df = re_index(df, keep_ticker=True)
        else: 
            df = re_index(df)
        dfs.append(df)

    merged_df = pd.concat(dfs, axis=1, join="outer", sort=True)
    merged_df.reset_index(inplace=True)

    merged_df.iloc[0] = merged_df.iloc[0].fillna(0)
    merged_df.loc[0,"Ticker"] = ticker
    merged_df.ffill(inplace=True)

    merged_df = remove_data_if_ticker_inactive(merged_df, ticker)
    merged_df = add_sector_data(merged_df, ticker)

    return merged_df 


def create_ML_feature_table():
    """
    Creates a table with all the features for the ML model.
    """
    
    spy_df, stock_tickers = load_SPY_components()
    all_trading_days = fetch_all_trading_days()

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()
    
    for ticker in tqdm(stock_tickers, desc="Creating ML Feature Table"):
        if check_if_close_price_exists(conn, "Normalized_Price_Data", ticker):
            
            data_dict, success = create_and_combine_node_feature_data(ticker)
            if not success:
                continue
            merged_df = align_and_merge_feature_data(data_dict, all_trading_days, ticker)

            create_and_upload_table( merged_df, table_name="ML_Feature_Table", column_names = merged_df.columns.tolist(),
                                                                                        primary_key = ["Ticker", "Date"]) 
            conn.commit()
        else:
            pass

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON ML_Feature_Table(Date);")
    conn.commit()

    conn.close()

if __name__ == "__main__":

    create_ML_feature_table()

