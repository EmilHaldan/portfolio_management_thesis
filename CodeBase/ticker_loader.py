import pandas as pd 
from datetime import datetime, timedelta
import tabulate
import json
import yfinance as yf
import pickle
import time 
import random as rn

def isin_dict(path = "Data/company_isin.json"):
    """
    Contains ISINs for each company name
    """
    isin_dict = json.load(open(path, "r"))
    isin_dict = {v: k for k, v in isin_dict.items()}
    return isin_dict



def load_close_prices(stock_ticker, start_date, end_date, time_frame = 15, interval = "1mo"):

    raise DeprecationWarning("This function is deprecated, use SQLite_tools instead ")
    stock_data = yf.download([stock_ticker], start_date, end_date , interval = interval)
    stock_data.reset_index(inplace=True)

    stock_data = stock_data[["Date", "Adj Close"]]
    stock_data['Returns'] = stock_data['Adj Close'].pct_change()
    stock_data.loc[0,'Returns'] = 0

    return stock_data


def load_SPY_components(path = "../Data/S&P500_historical_composition.csv"):
    
    df = pd.read_csv(path)

    df["Date"] = df["date"].apply(lambda x: datetime.strptime(str(x)[:10], "%Y-%m-%d"))

    all_dates = pd.date_range(start = df["Date"].min(), end = df["Date"].max(), freq = "D")
    all_dates = pd.DataFrame(all_dates, columns = ["Date"])

    df = pd.merge(all_dates, df, on = "Date", how = "left")
    df.ffill(inplace = True)

    df.drop(columns = ["date"], inplace = True)
    df.rename(columns = {"tickers": "Tickers"}, inplace = True)
    df = df[['Date', 'Tickers']]
    df["Number of Stocks"] = df["Tickers"].apply(lambda x: len(x.split(",")))
    df["Date"] = df["Date"].astype(str)
    df["Date"] = df["Date"].apply(lambda x: x[:10])

    df.reset_index(drop = True, inplace = True)

    all_stocks = list_all_stocks(df)

    # Randomizing the order of the stocks helping during debugging to find corner cases.
    # This was disabled further in the research process as the order of the stocks was important.
    # rn.shuffle(all_stocks)

    return df, all_stocks


def list_all_stocks(df_SPY_comp):
    """
    Returns a list of all stocks in the S&P500
    """
    all_stocks = []
    for row_idx, row in df_SPY_comp.iterrows():
        all_stocks += row["Tickers"].split(",")
    all_stocks = list(set(all_stocks))
    return all_stocks


def load_SPY_components_day(date : str = "1996-01-03", spy_df = None):
    """
    Load the S&P500 components for a given date
    """
    if spy_df is None:
        print("Please provide the spy_df, if you don't care, i'll do something inefficient just so it works")
        spy_df, _ = load_SPY_components()
    spy_df = spy_df[spy_df["Date"] == date]
    all_stocks = list_all_stocks(spy_df)

    return all_stocks



if __name__ == "__main__":

    # df = load_SPY_components()
    # print(df)

    all_stocks = load_SPY_components_day(date= "1996-01-03")
    print("all_stocks: ", all_stocks)
    print("")
    print("len(all_stocks): ", len(all_stocks))