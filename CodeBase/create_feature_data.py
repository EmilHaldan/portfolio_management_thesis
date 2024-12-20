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
from ticker_loader import load_SPY_components, load_SPY_components_day
import sqlite3
from tqdm import tqdm
import time 
import pickle

from SQLite_tools import query_stock_data, query_and_combine_TA_data, create_and_upload_table 
from SQLite_tools import check_if_close_price_exists, fetch_all_trading_days, get_all_quarters

pd.options.mode.chained_assignment = None 



def look_up_previous_mean_std(conn, date, feature, table_name = "Standardized_Price_Data"):
    cursor = conn.cursor()

    query = f"""
    SELECT * FROM {table_name}
    WHERE date = ?
    AND Feature = ?
    """
    try:
        cursor.execute(query, (date, feature))
        result = cursor.fetchall()
    except:
        return 0, 0, 0
    if len(result) == 0:
        return 0, 0, 0
    _date, _feature, mean, std, n = result[0][0], result[0][1], result[0][2], result[0][3], result[0][4]

    if n < 5:
        return 0, 0, 0
    else:
        return mean, std, n



def adjust_mean_std_n(old_mean, old_std, old_n, new_mean, new_std, new_n):
    """
    Adjust the mean, standard deviation, and n of a dataset.
    """
    n = old_n + new_n
    mean = (old_mean * old_n + new_mean * new_n) / n
    std = np.sqrt((old_n * (old_std ** 2 + old_mean ** 2) + new_n * (new_std ** 2 + new_mean ** 2) - n * mean ** 2) / n)

    return mean, std, n



def time_series_normalization_price_data():
    """
    This is done iteratively over time, to avoid information leakage from the future.
    """

    stock_data = query_stock_data(table="Stocks")
    stock_list = [x for x in stock_data["Ticker"].unique()]+ ["SPY"]
    all_spy_components_df, _ = load_SPY_components()

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()
    adj_table_name  = "Adjusted_Price_Data"
    standardized_table_name = "Standardized_Price_Data"
    norm_table_name = "Normalized_Price_Data"

    all_dates = fetch_all_trading_days()

    # Price data is transformed to be relative to the previous closing price, and uploadet to the database.
    # # NOTE: Disabled while testing
    for ticker in tqdm(stock_list, desc="Transforming and uploading price data"): 
        # Check if the ticker is already in the database. If it is, skip it.
        if check_if_close_price_exists(conn, adj_table_name, ticker):
            continue

        price_data = load_and_transform_price_data(ticker=ticker)
        create_and_upload_table(price_data, adj_table_name, column_names = ["Ticker","Date", "Open", "High", "Low", "Close", "Volume"],
                                                                                    primary_key = ["Ticker", "Date"])               

    previous_functional_date = all_dates[0]
    for idx,date in tqdm(enumerate(all_dates), total=len(all_dates), desc="Window normalizing price data step 1"):
        # Check if the date is a trading day
        if idx < 1:
            # Skipping the first entry since the prior transfomration is removed the first entry.
            continue

        spy_components_that_day = load_SPY_components_day(date = date, spy_df = all_spy_components_df)
        functional_date = date
        
        while len(spy_components_that_day) == 0:
            idx -= 1
            if idx <= 0:
                continue
            functional_date = all_dates[idx]

            spy_components_that_day = load_SPY_components_day(date = functional_date , spy_df = all_spy_components_df)

        adj_price_data = query_stock_data(table=adj_table_name, ticker="all_stocks", start_date=functional_date, end_date=functional_date)
        adj_price_data = adj_price_data[adj_price_data["Ticker"].isin(spy_components_that_day)]

        pivot_price_data = adj_price_data[["Date", "Ticker", "Close"]].pivot(index="Date", columns="Ticker", values="Close")

        values = pivot_price_data.values.flatten()
        values = values[~np.isnan(values)]

        old_mean, old_std, old_n = look_up_previous_mean_std(conn, previous_functional_date, feature="Close", table_name = standardized_table_name)

        new_mean = values.mean()
        new_std = values.std()
        new_n = len(values)

        mean, std, total_n = adjust_mean_std_n(old_mean, old_std, old_n, new_mean, new_std, new_n)

        norm_data = pd.DataFrame({"Date": [functional_date], "Feature": ["Close"], "Mean": [mean], "Std": [std], "N": [total_n]})

        previous_functional_date = functional_date

        create_and_upload_table(norm_data, standardized_table_name, column_names = ["Date", "Feature", "Mean", "Std", "N"], primary_key = ["Date", "Feature"])

        adj_price_data = adj_price_data[["Date", "Ticker", "Open", "High", "Low", "Close"]]

        adj_price_data["Open"] = (adj_price_data["Open"] - mean) / std
        adj_price_data["High"] = (adj_price_data["High"] - mean) / std
        adj_price_data["Low"] = (adj_price_data["Low"] - mean) / std
        adj_price_data["Close"] = (adj_price_data["Close"] - mean) / std

        create_and_upload_table(adj_price_data, norm_table_name, 
                                column_names = ["Date", "Ticker", "Open", "High", "Low", "Close"], 
                                primary_key = ["Date", "Ticker"])

        previous_functional_date = functional_date

    conn.close()



def load_and_transform_price_data(ticker: str, start_date= None, end_date= None):
    price_data = query_stock_data(table="Price_Data", ticker=ticker, start_date=start_date, end_date=end_date)
    previous_close = price_data["Close"].copy(deep=True)
    previous_close = previous_close.shift(1)

    price_data["Open"]  = (price_data["Open"]  / previous_close ) - 1
    price_data["High"]  = (price_data["High"]  / previous_close ) - 1
    price_data["Low"]   = (price_data["Low"]   / previous_close ) - 1
    price_data["Close"] = (price_data["Close"] / previous_close ) - 1
    price_data["Ticker"] = ticker

    price_data.dropna(inplace=True)

    return price_data


def load_and_transform_valuation_data(ticker: str, start_date= None, end_date= None):
    # The valuation data is transformed in this manner as the P/E, P/S, and P/B ratios are often used as a measure of how expensive a stock is.
    # The higher the value, the less attractive the stock is (Generally speaking). 
    # Thus, we take the inverse of the ratios to make the data more intuitive for ML processing
    # High values GOOD, Low values BAD, Close to 0 values VERY BAD

    valuation_data = query_stock_data(table="Valuation_Metrics", ticker=ticker, start_date=start_date, end_date=end_date)

    valuation_data["P_E_Ratio"] = 1 / valuation_data["P_E_Ratio"]
    valuation_data["P_S_Ratio"] = 1 / valuation_data["P_S_Ratio"]
    valuation_data["P_B_Ratio"] = 1 / valuation_data["P_B_Ratio"]

    # replace all values greater than 1 with 1.
    # We do this to combat outliers in the data, and a stock with a P/E ratio of less than 1 is often 
    # a sign of lack of interest in the company despite good fundamentals.
    valuation_data["P_E_Ratio"] = valuation_data["P_E_Ratio"].clip(upper=1, lower=-1)
    valuation_data["P_S_Ratio"] = valuation_data["P_S_Ratio"].clip(upper=1, lower=-1)
    valuation_data["P_B_Ratio"] = valuation_data["P_B_Ratio"].clip(upper=1, lower=-1)

    return valuation_data



def check_for_missing_quarterly_reports( stock_list: list, all_data: pd.DataFrame, activate_print: bool = False):
    """
    This function was used during development to check for missing quarterly reports in the fundamental data.
    """

    if activate_print:
        print("Checking for missing quarterly reports in the fundamental data.")
        # Iterate through each of the stocks in the stock_list and figure out how much data is missing for each stock.
        missing_data = {}

        for ticker in stock_list:
            specific_fundamentals = all_data[all_data["Ticker"] == ticker]

            # Iterate through each feature and see how much is missing for each feature for each stock.
            for feature in specific_fundamentals.columns:
                if feature not in missing_data:
                    if "Date" in feature or "Ticker" in feature:
                        continue
                    missing_data[feature] = {}
                
                missing_values = specific_fundamentals[feature].isnull().sum() / len(specific_fundamentals)
                # print("type(missing_values): ", type(missing_values))
                #if nan
                if missing_values >= 0:
                    missing_data[feature][ticker] = missing_values
                    # print(f"missing_data[{feature}][{ticker}] : ", missing_data[feature][ticker] )
        # print the distribution of missing data for each feature
        for feature in missing_data:
            print(f"Feature: {feature}")
            print("Mean missing data: ", np.mean(list(missing_data[feature].values())))
            print("Median missing data: ", np.median(list(missing_data[feature].values())))
            print("Max missing data: ", np.max(list(missing_data[feature].values())))
            print("Min missing data: ", np.min(list(missing_data[feature].values())))
            print("")     
    else: 
        pass


def fixing_filing_date_date_mismatches(all_data: pd.DataFrame, date_col = "Date"):
    """
    There are some rare cases where the date is before the filing date, which is impossible.
    These will be replaced with the filing date + 3 months to avoid any issues with the data.
    """
    
    # Ensure both columns are datetime
    all_data["EffectiveDate"] = pd.to_datetime(all_data["EffectiveDate"])
    all_data[date_col] = pd.to_datetime(all_data[date_col])

    # Identify mismatched rows and update the Date column
    mismatched = all_data[date_col] < all_data["EffectiveDate"]
    all_data.loc[mismatched, date_col] = all_data.loc[mismatched, "EffectiveDate"] + pd.DateOffset(months=3)

    # Convert back to string
    all_data[date_col] = all_data[date_col].dt.strftime("%Y-%m-%d")
    all_data["EffectiveDate"] = all_data["EffectiveDate"].dt.strftime("%Y-%m-%d")

    return all_data


def check_filing_date_consistency(merged_df, activate_print = False):
    """
    """
    if activate_print:
        df = merged_df.copy(deep=True)
        print("Checking for consistency in the FilingDate_x and FilingDate_y columns.")
        date_differences = []
        df["FilingDate_x"] = pd.to_datetime(df["FilingDate_x"])
        df["FilingDate_y"] = pd.to_datetime(df["FilingDate_y"])
        for idx, row in df.iterrows():
            diffalence = (row["FilingDate_x"] - row["FilingDate_y"])
            if type(diffalence) == pd._libs.tslibs.timedeltas.Timedelta:
                date_differences.append(diffalence.days)
            else: 
                pass
                # print("diffalence is not a timedelta object.")
                # print("diffalence: ", diffalence)
                # print("Ticker : ", row["Ticker"])
                # print("FilingDate_x : ", row["FilingDate_x"])
                # print("FilingDate_y : ", row["FilingDate_y"])
                # print("")

        print("Mean filing date difference: ", np.mean(date_differences))
        print("Median filing date difference: ", np.median(date_differences))
        print("Std filing date difference: ", np.std(date_differences))
        print("Max filing date difference: ", np.max(date_differences))
        print("Min filing date difference: ", np.min(date_differences))


def time_series_normalization_fundamental_data(activate_print = True):
    """
    This is done iteratively over time, to avoid information leakage from the future.
    """

    stock_data = query_stock_data(table="Stocks")
    stock_list = stock_data["Ticker"].unique()
    all_spy_components_df, _ = load_SPY_components()

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()
    adj_table_name  = "Adjusted_Fundamental_Data"
    standardized_table_name = "Standardized_Fundamental_Data"
    norm_table_name = "Normalized_Fundamental_Data"

    income_statements = query_stock_data(table="Income_Statements", ticker="all_stocks")
    balance_sheet = query_stock_data(table="Balance_Sheets", ticker="all_stocks")
    cash_flow = query_stock_data(table="Cash_Flow", ticker="all_stocks")

    # Replace FilingDate with Date for income_statement, balance_sheet, and earnings
    # FilingDate is the date of which the data becomes available, 
    # EffectiveDate reflects the date of the end of that quarter.
    # For the nromalization we want to use the EffectiveDate.
    # Missing values for FilingDate have been replaced with EffectiveDate + 3 months in the database creation process.

    income_statements = income_statements[["FilingDate", "EffectiveDate", "Ticker" ,"TotalRevenue", "GrossProfit", "EBIT", "EBITDA", "IncomeBeforeTax", "NetIncome"]]
    balance_sheet = balance_sheet[["FilingDate", "EffectiveDate", "Ticker", "TotalAssets", "TotalLiabilities"]]
    cash_flow = cash_flow[["FilingDate", "EffectiveDate", "Ticker","FreeCashFlow", "CapitalExpenditures"]]

    all_data = (income_statements.merge(balance_sheet, on=["Ticker", "EffectiveDate"], how="outer"))
    all_data = (all_data.merge(cash_flow, on=["Ticker", "EffectiveDate"], how="outer"))

    all_data = all_data.sort_values(by=["Ticker", "EffectiveDate"])
    check_for_missing_quarterly_reports(stock_list, all_data, activate_print = activate_print)  

    check_filing_date_consistency(all_data, activate_print = activate_print)

    # replace nan with "1800-01-01". This occours when 1 of the 3 dataframes are missing data for a specific quarter.
    # The below code selects the highest value for the FilingDate, as it is the most accurate.
    all_data["FilingDate"].fillna("1800-01-01", inplace=True)
    all_data["FilingDate_x"].fillna("1800-01-01", inplace=True)
    all_data["FilingDate_y"].fillna("1800-01-01", inplace=True)

    # Merging FilingDate, FilingDate_x, and FilingDate_y into a single column, and renaming it to Date.
    # The highest value for the Date column is used as it is the most accurate.
    all_data["Date"] = all_data[["FilingDate", "FilingDate_x", "FilingDate_y"]].max(axis=1)
    all_data.drop(columns=["FilingDate", "FilingDate_x", "FilingDate_y"], inplace=True)

    # Making super sure that the Date column is before the EffectiveDate column, as it should be.
    all_data = fixing_filing_date_date_mismatches(all_data)

    all_data = all_data[["Date", "EffectiveDate", "Ticker", "TotalRevenue", "GrossProfit", "EBIT", 
                        "EBITDA", "IncomeBeforeTax", "NetIncome", "TotalAssets", "TotalLiabilities", "FreeCashFlow", "CapitalExpenditures"]]

    # Drops a row if more than 5 values are missing in the row.
    all_data.dropna(thresh=len(all_data.columns) - 5, inplace=True)
    all_data.reset_index(drop=True, inplace=True)

    # Forward fill the data to avoid missing values in the relative data and fill the rest with 0.
    # Forward fill for each stock (Ticker)
    all_data = all_data.groupby("Ticker").apply(lambda group: group.ffill())
    all_data = all_data.fillna(0)
    all_data.reset_index(drop=True, inplace=True)

    # We Replace all 0's with 1's to avoid division by zero in future transformations. Doing so won't make a difference in the normalization.
    all_data = all_data.replace(0, 1)


    relative_df = all_data.copy(deep=True)
    relative_df_dates = relative_df[["Date", "EffectiveDate"]].copy(deep=True)
    relative_df.drop(columns=["Date", "EffectiveDate"], inplace=True)
    for col in [x for x in relative_df.columns if x != "Ticker"]:
        relative_df[col] = relative_df.groupby("Ticker")[col].pct_change()
        relative_df[col].iloc[relative_df.groupby("Ticker").head(1).index] = 0

    value_columns = [x for x in relative_df.columns if x != "Ticker"]

    # We clip the values to be between -1 and 1 to avoid abnormal outliers in the data. 
    relative_df[value_columns] = relative_df[value_columns].clip(lower=-1, upper=1)

    relative_df.columns = ["Ticker"] + ["Relative_" + column for column in relative_df.columns if column != "Ticker"]
    relative_df = pd.concat([relative_df_dates, relative_df], axis=1)
    # relative_df.to_excel(f"../Data/fundamental_relative_df.xlsx")

    # NOTE: Cube Root Transformations are used to scale the magnitude of the data. Log10 was not possible due to negative values.
    cbrt_df = all_data.copy(deep=True)
    cbrt_df_dates = cbrt_df[["Date", "EffectiveDate", "Ticker"]].copy(deep=True)
    cbrt_df.drop(columns=["Date", "Ticker", "EffectiveDate"], inplace=True)
    
    # Cube root transformation of the data
    cbrt_df = np.cbrt(cbrt_df)
    cbrt_df = np.cbrt(cbrt_df)
    cbrt_df.columns = ["Cbrt_" + column for column in cbrt_df.columns]
    cbrt_df = pd.concat([cbrt_df_dates, cbrt_df], axis=1)

    transformed_fundamentals = all_data.merge(relative_df, on=["Date", "EffectiveDate", "Ticker"], how="outer")
    transformed_fundamentals = transformed_fundamentals.merge(cbrt_df, on=["Date", "EffectiveDate", "Ticker"], how="outer")
    # transformed_fundamentals.to_excel(f"../Data/fundamental_merged_data.xlsx")
    # Remove original columns from the dataframe as they are no longer of use.
    transformed_fundamentals.drop(columns=["TotalRevenue", "GrossProfit", "EBIT", 
                                        "EBITDA", "IncomeBeforeTax", "NetIncome", 
                                        "TotalAssets", "TotalLiabilities", "FreeCashFlow", "CapitalExpenditures"], inplace=True)

    # Upload the transformed fundamental data to the database.
    create_and_upload_table(transformed_fundamentals, adj_table_name, column_names = ["Date", "EffectiveDate", "Ticker",
                        "Relative_TotalRevenue", "Relative_GrossProfit", "Relative_EBIT", "Relative_EBITDA", 
                        "Relative_IncomeBeforeTax", "Relative_NetIncome", "Relative_TotalAssets", "Relative_TotalLiabilities",
                        "Relative_FreeCashFlow", "Relative_CapitalExpenditures",
                        "Cbrt_TotalRevenue", "Cbrt_GrossProfit", "Cbrt_EBIT", 
                        "Cbrt_EBITDA", "Cbrt_IncomeBeforeTax", "Cbrt_NetIncome", "Cbrt_TotalAssets", "Cbrt_TotalLiabilities",
                        "Cbrt_FreeCashFlow", "Cbrt_CapitalExpenditures"],
                        primary_key = ["Date", "EffectiveDate", "Ticker"])

    # We will normalize the fundamental data, with a rolling window of 16 quarter, 4 years, to account for the seasonality of the data,
    # Financial crises, and other events that might affect the data.
    all_quarter_dates = transformed_fundamentals["EffectiveDate"].unique()
    previous_date = fetch_all_trading_days()[0]
    
    for idx, date in tqdm(enumerate(all_quarter_dates), total=len(all_quarter_dates), desc="Window normalizing fundamental data"):
        spy_components_that_day = load_SPY_components_day(date = date, spy_df = all_spy_components_df)

        if idx >= 16:
            previous_date = all_quarter_dates[idx-16]
        local_transformed_fundamentals = transformed_fundamentals[(transformed_fundamentals["EffectiveDate"] >= previous_date) 
                                                                  & (transformed_fundamentals["EffectiveDate"] <= date)]

        for column in [x for x in local_transformed_fundamentals.columns if x not in ["Ticker", "Date", "EffectiveDate"]]:
            values = local_transformed_fundamentals[column].values
            values = values[~np.isnan(values)]

            mean = values.mean()
            std = values.std()
            n = len(values)

            if n < 5:
                continue

            norm_data = pd.DataFrame({"Date": [date], "Feature": [column], "Mean": [mean], "Std": [std], "N": [n]})

            create_and_upload_table(norm_data, standardized_table_name , column_names = ["Date", "Feature", "Mean", "Std", "N"], primary_key = ["Date", "Feature"])

            local_transformed_fundamentals[column] = (local_transformed_fundamentals[column] - mean) / std

        local_transformed_fundamentals.fillna(0, inplace=True)
        create_and_upload_table(local_transformed_fundamentals, norm_table_name, 
                                column_names = local_transformed_fundamentals.columns, primary_key = ["Date", "EffectiveDate", "Ticker"])

    conn.close()



def time_series_normalization_earnings_data(activate_print = True):
    """
    This is done iteratively over time, to avoid information leakage from the future.
    """

    stock_data = query_stock_data(table="Stocks")
    stock_list = stock_data["Ticker"].unique()
    all_spy_components_df, _ = load_SPY_components()

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()
    adj_table_name  = "Adjusted_Earnings_Data"
    standardized_table_name = "Standardized_Earnings_Data"
    norm_table_name = "Normalized_Earnings_Data"

    earnings = query_stock_data(table="Earnings", ticker="all_stocks")
    earnings = earnings[["FilingDate", "EffectiveDate", "Ticker", "EPSActual", "EPSDifference"]]
    earnings.rename(columns={"FilingDate": "Date"}, inplace=True)

    # If EPSDifference is missing, let it be equal to EPSActual, as the EPSEstimate is missing, thus 0.
    earnings["EPSDifference"] = earnings["EPSDifference"].fillna(earnings["EPSActual"])

    missing_values = earnings.isnull().sum().sum()
    if missing_values and activate_print:
        print("Checking for missing values in the earnings data.")
        print(missing_values, " in the dataset")
        print("")


    # # # Upload the transformed fundamental data to the database.
    create_and_upload_table(earnings, adj_table_name, column_names = ["Date", "EffectiveDate", "Ticker",
                        "EPSActual", "EPSDifference"],
                        primary_key = ["Date", "EffectiveDate", "Ticker"])

    # # # Normalize the data using a rolling window fashion similar to the price data, but with intervals of 1 quarter.
    # # We will normalize the fundamental data, with a rolling window of 16 quarter, 4 years, to account for the seasonality of the data,
    # # Financial crises, and other events that might affect the data.
    all_quarter_dates = earnings["EffectiveDate"].unique()
    previous_date = fetch_all_trading_days()[0]
    
    for idx, date in tqdm(enumerate(all_quarter_dates), total=len(all_quarter_dates), desc="Window normalizing earnings data part 1"):
        spy_components_that_day = load_SPY_components_day(date = date, spy_df = all_spy_components_df)

        if idx >= 16:
            previous_date = all_quarter_dates[idx-16]
        local_earnings = earnings[(earnings["EffectiveDate"] >= previous_date) 
                                                                  & (earnings["EffectiveDate"] <= date)]

        for column in [x for x in local_earnings.columns if x not in ["Ticker", "Date", "EffectiveDate"]]:
            values = local_earnings[column].values
            values = values[~np.isnan(values)]

            mean = values.mean()
            std = values.std()
            n = len(values)

            if n < 5:
                continue

            # Create a dataframe with the functional date, feature_name, the mean, and the standard deviation
            norm_data = pd.DataFrame({"Date": [date], "Feature": [column], "Mean": [mean], "Std": [std], "N": [n]})

            create_and_upload_table(norm_data, standardized_table_name , column_names = ["Date", "Feature", "Mean", "Std", "N"], primary_key = ["Date", "Feature"])


    previous_date = fetch_all_trading_days()[0]
    for idx, date in tqdm(enumerate(all_quarter_dates), total=len(all_quarter_dates), desc="Window normalizing earnings data part 2"):
        spy_components_that_day = load_SPY_components_day(date = date, spy_df = all_spy_components_df)

        local_earnings = earnings[(earnings["EffectiveDate"] >= previous_date) 
                                        & (earnings["EffectiveDate"] <= date)]

        for column in [x for x in local_earnings.columns if x not in ["Ticker", "Date", "EffectiveDate"]]:
            
            mean, std, n = look_up_previous_mean_std(conn, date, feature=column, table_name = standardized_table_name)
            if n < 5:
                break
            
            local_earnings[column] = (local_earnings[column] - mean) / std

        if n < 5:
                continue
        local_earnings.fillna(0, inplace=True)
        create_and_upload_table(local_earnings, norm_table_name, 
                                column_names = local_earnings.columns, primary_key = ["Date", "EffectiveDate", "Ticker"])

        previous_date = date

    conn.close()



def time_series_normalization_valuation_data():
    """
    This is done iteratively over time, to avoid information leakage from the future.
    """

    stock_data = query_stock_data(table="Stocks")
    stock_list = stock_data["Ticker"].unique()
    all_spy_components_df, _ = load_SPY_components()

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()
    adj_table_name  = "Adjusted_Valuation_Data"
    standardized_table_name = "Standardized_Valuation_Data"
    norm_table_name = "Normalized_Valuation_Data"

    all_dates = fetch_all_trading_days()

    for ticker in tqdm(stock_list, desc="Transforming and uploading valutaion data"): 
        # Check if the ticker is already in the database. If it is, skip it.
        if check_if_close_price_exists(conn, adj_table_name, ticker):
            continue

        valuation_data = load_and_transform_valuation_data(ticker=ticker)
        create_and_upload_table(valuation_data, adj_table_name, column_names = ["Date","Ticker", "P_E_Ratio", "P_S_Ratio", "P_B_Ratio"],
                                                                                        primary_key = ["Date", "Ticker"])

    previous_functional_date = all_dates[0]
    for idx,date in tqdm(enumerate(all_dates), total=len(all_dates), desc="Window normalizing valutation data"):
        if idx < 1:
            continue

        spy_components_that_day = load_SPY_components_day(date = date, spy_df = all_spy_components_df)
        functional_date = date
        
        while len(spy_components_that_day) == 0:
            idx -= 1
            if idx <= 0:
                continue
            functional_date = all_dates[idx]

            spy_components_that_day = load_SPY_components_day(date = functional_date , spy_df = all_spy_components_df)

        adj_valuation_data = query_stock_data(table=adj_table_name, ticker="all_stocks", start_date=functional_date, end_date=functional_date)
        adj_valuation_data = adj_valuation_data[adj_valuation_data["Ticker"].isin(spy_components_that_day)]

        for feature in ["P_E_Ratio", "P_S_Ratio", "P_B_Ratio"]:
            pivot_valuation_data = adj_valuation_data[["Date", "Ticker", feature]].pivot(index="Date", columns="Ticker", values=feature)

            values = pivot_valuation_data.values.flatten()
            values = values[~np.isnan(values)]

            old_mean, old_std, old_n = look_up_previous_mean_std(conn, previous_functional_date, feature=feature, table_name = standardized_table_name)

            new_mean = values.mean()
            new_std = values.std()
            new_n = len(values)

            mean, std, total_n = adjust_mean_std_n(old_mean, old_std, old_n, new_mean, new_std, new_n)

            norm_data = pd.DataFrame({"Date": [functional_date], "Feature": [feature], "Mean": [mean], "Std": [std], "N": [total_n]})

            previous_functional_date = functional_date

            create_and_upload_table(norm_data, standardized_table_name, column_names = ["Date", "Feature", "Mean", "Std", "N"], primary_key = ["Date", "Feature"])
            
            adj_valuation_data[feature] = (adj_valuation_data[feature] - mean) / std

        adj_valuation_data.fillna(0, inplace=True)
        
        create_and_upload_table(adj_valuation_data, norm_table_name,
                                column_names = ["Date", "Ticker", "P_E_Ratio", "P_S_Ratio", "P_B_Ratio"],
                                primary_key = ["Date", "Ticker"])

    conn.close()




def time_series_normalization_TA_data():
    """
    This is done iteratively over time, to avoid information leakage from the future.
    """

    stock_data = query_stock_data(table="Stocks")
    stock_list = stock_data["Ticker"].unique()
    all_spy_components_df, _ = load_SPY_components()

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()
    standardized_table_name = "Standardized_TA_Data"
    norm_table_name = "Normalized_TA_Data"

    all_dates = fetch_all_trading_days()

    # No prior transformation needed for TA, as they are already transformed in their desired format during creating the financial features. 
    # just normalization needed.

    previous_functional_date = all_dates[1]
    for idx,date in tqdm(enumerate(all_dates), total=len(all_dates), desc="Window normalizing TA data"):
        if idx < 2:
            # We skip 2 because MOST of the TA data is missing for the first days worth of data.
            continue

        spy_components_that_day = load_SPY_components_day(date = date, spy_df = all_spy_components_df)
        functional_date = date
        
        while len(spy_components_that_day) == 0:
            idx -= 1
            if idx <= 0:
                continue
            functional_date = all_dates[idx]

            spy_components_that_day = load_SPY_components_day(date = functional_date, spy_df = all_spy_components_df)

        all_TA_data = query_and_combine_TA_data(ticker="all_stocks", start_date=functional_date, end_date=functional_date)
        all_TA_data
        for feature in ["Max_High_10", "Min_Low_10", "Max_High_20", "Min_Low_20", 
                        "AROON_down_10", "AROON_up_10", "AROON_down_20", "AROON_up_20",
                        "MACD_10", "MACD_20", "MACD_signal_10", "MACD_signal_20",
                        "MACD_hist_10", "MACD_hist_20", "RSI_10", "RSI_20",
                        "EMA_10", "EMA_20", "ROC_10", "ROC_20",
                        "BB_upper_10", "BB_lower_10", "BB_upper_20", "BB_lower_20"]:

            specific_ta_data = all_TA_data[["Date", "Ticker", feature]]
            pivot_valuation_data = specific_ta_data[["Date", "Ticker", feature]].pivot(index="Date", columns="Ticker", values=feature)

            values = pivot_valuation_data.values.flatten()
            for special_feature in ["AROON_down_10", "AROON_up_10", "AROON_down_20", "AROON_up_20",
                                    "RSI_10", "RSI_20"]:
                old_mean, old_std, old_n = look_up_previous_mean_std(conn, previous_functional_date, feature=feature, table_name = standardized_table_name)

                new_mean = values.mean()
                new_std = values.std()
                new_n = len(values)

                mean, std, total_n = adjust_mean_std_n(old_mean, old_std, old_n, new_mean, new_std, new_n)

                norm_data = pd.DataFrame({"Date": [functional_date], "Feature": [feature], "Mean": [mean], "Std": [std], "N": [total_n]})
                create_and_upload_table(norm_data, standardized_table_name, column_names = ["Date", "Feature", "Mean", "Std", "N"], primary_key = ["Date", "Feature"])

                all_TA_data[feature] = (all_TA_data[feature] - mean) / std
            
        previous_functional_date = functional_date

        all_TA_data.fillna(0, inplace=True)

        create_and_upload_table(all_TA_data, norm_table_name,
                                column_names = ["Date", "Ticker", "Max_High_10", "Min_Low_10", "Max_High_20", "Min_Low_20", 
                                                "AROON_down_10", "AROON_up_10", "AROON_down_20", "AROON_up_20",
                                                "MACD_10", "MACD_20", "MACD_signal_10", "MACD_signal_20",
                                                "MACD_hist_10", "MACD_hist_20", "RSI_10", "RSI_20",
                                                "EMA_10", "EMA_20", "ROC_10", "ROC_20",
                                                "BB_upper_10", "BB_lower_10", "BB_upper_20", "BB_lower_20"],
                                primary_key = ["Date", "Ticker"])
    conn.close()


def create_1_hot_encoding_for_sectors():

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()

    table_name = "Sector_1hot_encoded"

    stock_data = query_stock_data(table="Stocks")
    
    stock_data = stock_data[["Ticker", "Sector"]]

    # rerepresenting the dummies as 0 and 1's
    stock_data = pd.get_dummies(stock_data, columns=["Sector"], prefix="sector")
    stock_data.columns = [col.replace("sector_", "") for col in stock_data.columns]
    stock_data.columns = [col.replace(" ", "_") for col in stock_data.columns]
    stock_data = stock_data.groupby("Ticker").sum()
    stock_data.reset_index(inplace=True)


    create_and_upload_table(stock_data, table_name, column_names = stock_data.columns, primary_key = ["Ticker"])

    conn.close()


def create_active_stock_table():
    """
    This function shall read the S&P500 composition, and create a table with 3 attributes (Date, Ticker, Active)
    The Active attribute will be 1 if the stock is present in the S&P500 for that date, and 0 otherwise.
    """

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()

    table_name = "Active_Stocks"
    all_spy_components_df, all_stock_tickers = load_SPY_components()
    active_trade_days = fetch_all_trading_days()

    active_stocks_rows = []
    for idx, row in tqdm(all_spy_components_df.iterrows(), total=len(all_spy_components_df), desc="Creating active stock table as a mask for ML data"):
        date = row["Date"]
        if date not in active_trade_days:
            continue
        active_tickers = row["Tickers"].split(",")
        
        for stock in all_stock_tickers:
            if stock in active_tickers:
                active = 1
            else:
                active = 0
            active_stocks_rows.append([date, stock, active])

    active_stocks = pd.DataFrame(active_stocks_rows, columns=["Date", "Ticker", "Active"])
    create_and_upload_table(active_stocks, table_name, column_names = active_stocks.columns, primary_key = ["Date", "Ticker"])
    conn.close()


def run_time_pr_section(start_time, section_name):
    print("\n"*1)
    print(f"Time for {section_name}: {time.time()-start_time}")
    print("\n"*1)
    return time.time()


def run_all_normalization_processes():

    start_time = time.time()
    create_1_hot_encoding_for_sectors()
    run_time_pr_section(start_time, "1-Hot Encoding for Sectors")

    start_time = time.time()
    create_active_stock_table()
    run_time_pr_section(start_time, "Active Stock Table Creation")

    start_time = time.time()
    time_series_normalization_price_data()
    run_time_pr_section(start_time, "Price Data Normalization")

    start_time = time.time()
    time_series_normalization_fundamental_data()
    run_time_pr_section(start_time, "Fundamental Data Normalization")
    
    start_time = time.time()
    time_series_normalization_earnings_data()
    run_time_pr_section(start_time, "Earnings Data Normalization")
    
    start_time = time.time()
    time_series_normalization_valuation_data()
    run_time_pr_section(start_time, "Valuation Data Normalization")
    
    start_time = time.time()
    time_series_normalization_TA_data()
    run_time_pr_section(start_time, "TA Data Normalization")
    


if __name__ == "__main__":

    run_all_normalization_processes()