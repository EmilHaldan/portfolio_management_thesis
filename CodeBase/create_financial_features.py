import pandas as pd
import numpy as np
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
from datetime import datetime
import talib

from create_financial_database import get_credentials, populate_database
from SQLite_tools import query_stock_data, create_and_upload_table, check_if_close_price_exists
import time

# supression of pandas warnings
pd.options.mode.chained_assignment = None  


def create_and_update_valuation_data(start_date="1996-01-01", end_date="2024-07-01"):
    """
    Creates all valuation data for all stocks in the database.
    """
 
    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()

    spy_df, stock_tickers = load_SPY_components()

    for ticker in tqdm(stock_tickers, desc="Creating valuation metrics for all stocks"):
        price_df = query_stock_data(table="Price_Data", ticker=ticker, start_date = start_date, end_date = end_date)
        income_statement_df = query_stock_data(table="Income_Statements", ticker=ticker, start_date = start_date, end_date = end_date)
        balance_sheet_df = query_stock_data(table="Balance_Sheets", ticker=ticker, start_date = start_date, end_date = end_date)
        earnings_df = query_stock_data(table="Earnings", ticker=ticker, start_date = start_date, end_date = end_date)

        if (price_df.empty) or (income_statement_df.empty) or (balance_sheet_df.empty) or (earnings_df.empty):
            continue

        financial_ratios_df = price_df[["Ticker", "Date", "Close", "Adjusted_Scalar"]].copy()
        financial_ratios_df["Date"] = pd.to_datetime(financial_ratios_df["Date"])

        earnings_df["Date"] = pd.to_datetime(earnings_df["FilingDate"])
        income_statement_df["Date"] = pd.to_datetime(income_statement_df["FilingDate"])
        balance_sheet_df["Date"] = pd.to_datetime(balance_sheet_df["FilingDate"])

        financial_ratios_df = pd.merge(financial_ratios_df, earnings_df[["Date", "EPSActual"]], on ="Date", how="left")
        financial_ratios_df = pd.merge(financial_ratios_df, income_statement_df[["Date", "NetIncome", "TotalRevenue"]], on ="Date", how="left")
        financial_ratios_df = pd.merge(financial_ratios_df, balance_sheet_df[["Date", "TotalAssets", "TotalLiabilities", "OutstandingShares"]], on ="Date", how="left")
        
        financial_ratios_df = financial_ratios_df.ffill()
        
        financial_ratios_df["P_E_Ratio"] = financial_ratios_df["Close"] / financial_ratios_df["EPSActual"]
        financial_ratios_df["P_S_Ratio"] = financial_ratios_df["Close"] / (financial_ratios_df["TotalRevenue"] / financial_ratios_df["OutstandingShares"])
        financial_ratios_df["P_B_Ratio"] = financial_ratios_df["Close"] / ((financial_ratios_df["TotalAssets"] - financial_ratios_df["TotalLiabilities"]) / financial_ratios_df["OutstandingShares"])
        
        financial_ratios_df["P_E_Ratio"] = financial_ratios_df["P_E_Ratio"].clip(upper=200, lower=-200)
        financial_ratios_df["P_S_Ratio"] = financial_ratios_df["P_S_Ratio"].clip(upper=200, lower=-200)
        financial_ratios_df["P_B_Ratio"] = financial_ratios_df["P_B_Ratio"].clip(upper=200, lower=-200)
        
        financial_ratios_df[["P_E_Ratio", "P_S_Ratio", "P_B_Ratio"]] = financial_ratios_df[["P_E_Ratio", "P_S_Ratio", "P_B_Ratio"]].fillna(0.0)
        
        financial_ratios_df = financial_ratios_df[["Date", "Ticker", "P_E_Ratio", "P_S_Ratio", "P_B_Ratio"]]
        financial_ratios_df["Date"] = financial_ratios_df["Date"].dt.strftime('%Y-%m-%d')

        create_and_upload_table(financial_ratios_df, table_name=f"Valuation_Metrics", column_names=["Date", "Ticker", "P_E_Ratio", "P_S_Ratio", "P_B_Ratio"])
        
        conn.commit()
        
    conn.close()
         
    return None



def create_and_update_technical_indicator_data(start_date="1996-01-01", end_date="2024-07-01"):
    """
    Creates all sets of technical indicators for all stocks in the database.
    """
 
    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()

    spy_df, stock_tickers = load_SPY_components()

    for ticker in tqdm(stock_tickers, desc="Creating technical indicators for all stocks"):
        df = query_stock_data(table="Price_Data", ticker=ticker, start_date = start_date, end_date = end_date)

        if df.empty:
            continue

        for timeperiod in [10, 20]:
            aroon_df = aroon(df, ticker, timeperiod)
            bollinger_bands_df = bollinger_bands(df, ticker, timeperiod)
            ema_df = ema(df, ticker, timeperiod)
            macd_df = macd(df, ticker, timeperiod)
            roc_df = roc(df, ticker, timeperiod)
            rsi_df = rsi(df, ticker, timeperiod)
            max_high_df = max_high(df, ticker, timeperiod)
            min_low_df = min_low(df, ticker, timeperiod)
            
            # Upload technical indicators to the database
            create_and_upload_table(aroon_df, table_name=f"AROON_{timeperiod}", column_names=["Date", "Ticker", f"AROON_down_{timeperiod}", f"AROON_up_{timeperiod}"])
            create_and_upload_table(bollinger_bands_df, table_name=f"Bollinger_Bands_{timeperiod}", column_names=["Date", "Ticker", f"BB_upper_{timeperiod}", f"BB_lower_{timeperiod}"])
            create_and_upload_table(ema_df, table_name=f"EMA_{timeperiod}", column_names=["Date", "Ticker", f"EMA_{timeperiod}"])
            create_and_upload_table(macd_df, table_name=f"MACD_{timeperiod}", column_names=["Date", "Ticker", f"MACD_{timeperiod}", f"MACD_signal_{timeperiod}", f"MACD_hist_{timeperiod}"])
            create_and_upload_table(roc_df, table_name=f"ROC_{timeperiod}", column_names=["Date", "Ticker", f"ROC_{timeperiod}"])
            create_and_upload_table(rsi_df, table_name=f"RSI_{timeperiod}", column_names=["Date", "Ticker", f"RSI_{timeperiod}"])
            create_and_upload_table(max_high_df, table_name=f"Max_High_{timeperiod}", column_names=["Date", "Ticker", f"Max_High_{timeperiod}"])
            create_and_upload_table(min_low_df, table_name=f"Min_Low_{timeperiod}", column_names=["Date", "Ticker", f"Min_Low_{timeperiod}"])

    conn.close()
         
    return None


def aroon(df, ticker, timeperiod=10):
    aroon_down, aroon_up = talib.AROON(df['High'], df['Low'], timeperiod=timeperiod)
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df["Ticker"] = ticker
    new_df[f'AROON_down_{timeperiod}'] = aroon_down
    new_df[f'AROON_up_{timeperiod}'] = aroon_up
    # Missing values for AROON can be replaced by 50, as the span varies between 0 and 100
    new_df.fillna(50, inplace=True)

    return new_df


def bollinger_bands(df, ticker, timeperiod=10, nbdevup=2, nbdevdn=2):
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn)
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df["Ticker"] = ticker
    new_df[f'BB_upper_{timeperiod}'] = upper.where(upper.notna(), df['Close'])
    new_df[f'BB_lower_{timeperiod}'] = lower.where(lower.notna(), df['Close'])

    # Bands are set relative to the Close price 
    new_df[f'BB_upper_{timeperiod}'] = (new_df[f'BB_upper_{timeperiod}'] - df['Close'])  / df['Close']
    new_df[f'BB_lower_{timeperiod}'] = (new_df[f'BB_lower_{timeperiod}'] - df['Close'])  / df['Close']

    new_df.fillna(0, inplace=True)

    return new_df


def ema(df, ticker, timeperiod=10):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df["Ticker"] = ticker
    new_df[f'EMA_{timeperiod}'] = talib.EMA(df['Close'], timeperiod=timeperiod)
    new_df[f'EMA_{timeperiod}'] = new_df[f'EMA_{timeperiod}'].where(new_df[f'EMA_{timeperiod}'].notna(), df['Close'])

    # EMA is set relative to the Close price
    new_df[f'EMA_{timeperiod}'] = (new_df[f'EMA_{timeperiod}'] - df['Close']) / df['Close']

    new_df.fillna(0, inplace=True)

    return new_df


def macd(df, ticker, timeperiod=9):

    fastperiod=round(timeperiod*1.333)  # usually static at 12
    slowperiod=round(timeperiod*2.888)  # usually static at 26
    signalperiod=timeperiod             # usually static at 9

    macd, signal, hist = talib.MACD(df['Close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)

    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df["Ticker"] = ticker

    # Define 2 price normalized EMAs as described in the MACD transformation in the report.
    new_df[f'EMA_{fastperiod}'] = ema(df, ticker, fastperiod)[f'EMA_{fastperiod}']
    new_df[f'EMA_{slowperiod}'] = ema(df, ticker, slowperiod)[f'EMA_{slowperiod}']

    new_df[f'MACD_{timeperiod}'] = new_df[f'EMA_{fastperiod}'] - new_df[f'EMA_{slowperiod}']
    new_df[f'MACD_signal_{timeperiod}'] = talib.EMA(new_df[f'MACD_{timeperiod}'], timeperiod=signalperiod)

    new_df[f'MACD_hist_{timeperiod}'] = new_df[f'MACD_{timeperiod}'] - new_df[f'MACD_signal_{timeperiod}']

    # Missing values for MACD can be replaced by 0
    new_df.fillna(0, inplace=True)

    return new_df


def roc(df, ticker,  timeperiod=10):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df["Ticker"] = ticker
    new_df[f'ROC_{timeperiod}'] = talib.ROC(df['Close'], timeperiod=timeperiod)
    new_df[f'ROC_{timeperiod}'] = new_df[f'ROC_{timeperiod}'].clip(upper=100)

    # Missing values for ROC can be replaced by 0
    new_df.fillna(0, inplace=True)

    return new_df


def rsi(df, ticker, timeperiod=10):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df["Ticker"] = ticker
    new_df[f'RSI_{timeperiod}'] = talib.RSI(df['Close'], timeperiod=timeperiod)

    # Missing values for RSI can be replaced by 50, as the span varies between 0 and 100
    new_df.fillna(50, inplace=True)

    # Scale and shift the RSI values between -1 and 1 as described in the report.
    new_df[f'RSI_{timeperiod}'] = (new_df[f'RSI_{timeperiod}'] - 50) / 50

    return new_df


def max_high(df, ticker, timeperiod=10):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df["Ticker"] = ticker
    new_df[f'Max_High_{timeperiod}'] = talib.MAX(df['High'], timeperiod=timeperiod)
    new_df[f'Max_High_{timeperiod}'] = (new_df[f'Max_High_{timeperiod}'] - df['Close']) / df['Close']

    new_df.fillna(0, inplace=True)

    return new_df


def min_low(df, ticker, timeperiod=10):
    new_df = pd.DataFrame()
    new_df['Date'] = df['Date']
    new_df["Ticker"] = ticker
    new_df[f'Min_Low_{timeperiod}'] = talib.MIN(df['Low'], timeperiod=timeperiod)
    new_df[f'Min_Low_{timeperiod}'] = (new_df[f'Min_Low_{timeperiod}'] - df['Close']) / df['Close']

    new_df.fillna(0, inplace=True)

    return new_df


def test_creation_of_ta():
    
    ticker = "NVDA"
    df = query_stock_data(table="Price_Data", ticker="NVDA", start_date="2016-01-01", end_date="2024-10-31")
    df

    print("df:")
    print(df.head())

    # create tests for the TA functions
    print("head of original df")
    print(df.head())
    print("\n tail of original df")
    print(df.tail())
    print("\n"*2)

    tmp_df = aroon(df, ticker)
    print("head of aroon")
    print(tmp_df.head())
    print("\n tail of aroon")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = bollinger_bands(df , ticker)
    print("head of bollinger_bands")
    print(tmp_df.head(30))
    print("\n tail of bollinger_bands")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = ema(df , ticker)
    print("head of ema")
    print(tmp_df.head())
    print("\n tail of ema")
    print(tmp_df.tail())
    print("\n"*2)
    
    tmp_df = macd(df , ticker)
    print("head of macd")
    print(tmp_df.head())
    print("\n tail of macd")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = roc(df , ticker)
    print("head of roc")
    print(tmp_df.head())
    print("\n tail of roc")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = rsi(df , ticker)
    print("head of rsi")
    print(tmp_df.head())
    print("\n tail of rsi")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = max_high(df , ticker)
    print("head of max_high")
    print(tmp_df.head(20))
    print("\n tail of max_high")
    print(tmp_df.tail())
    print("\n"*2)

    tmp_df = min_low(df , ticker)
    print("head of min_low")
    print(tmp_df.head(20))
    print("\n tail of min_low")
    print(tmp_df.tail())
    print("\n"*2)


if __name__ == "__main__":


    create_and_update_technical_indicator_data()
    create_and_update_valuation_data()

    # test_creation_of_ta()













    

