
import sqlite3
import pandas as pd
from tqdm import tqdm
import pickle
import os


def query_stock_data(table="Price_Data", ticker= "all_stocks", start_date = None, end_date = None):
    """
    Query stock data from an SQLite database.

    Parameters:
    db_name (str): The name or path of the SQLite database file.
    table (str): The name of the table to query.
    ticker (str): The stock ticker symbol to filter by.
    start_date (str): The start date for the query in the format 'YYYY-MM-DD'.
    end_date (str): The end date for the query in the format 'YYYY-MM-DD'.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the query results.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect("../Data/EOD_financial_data.db")

    # Create a SQL query string
    if start_date == None or end_date == None:
        start_date = "1996-01-01"
        end_date = "2024-07-02"
    if table in ["Balance_Sheets", "Cash_Flow", "Income_Statements", "Earnings"]:
        if ticker.lower() == "all_stocks":
            query = f"""
            SELECT * FROM {table}
            WHERE FilingDate BETWEEN ? AND ?
            ORDER BY FilingDate;
            """
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))

        else:
            query = f"""
            SELECT * FROM {table}
            WHERE Ticker = ?
            AND FilingDate BETWEEN ? AND ?
            ORDER BY FilingDate;
            """
            df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))

    elif table in ["Adjusted_Fundamental_Data" ,"Normalized_Fundamental_Data",
            "Adjusted_Earnings_Data", "Normalized_Earnings_Data"]:
        if ticker.lower() == "all_stocks":
            query = f"""
            SELECT * FROM {table}
            WHERE Date BETWEEN ? AND ?
            ORDER BY Date;
            """
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        else:
            query = f"""
            SELECT * FROM {table}
            WHERE Ticker = ?
            AND Date BETWEEN ? AND ?
            ORDER BY Date;
            """
            df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date))

    elif table in ["Price_Data" , "Adjusted_Price_Data" ,"Normalized_Price_Data",
                    "Valuation_Metrics", "Adjusted_Valuation_Metrics", "Normalized_Valuation_Metrics"]:
        if ticker.lower() == "all_stocks":
            query = f"""
            SELECT * FROM {table}
            WHERE date BETWEEN ? AND ?
            ORDER BY date;
            """
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))

        else:
            query = f"""
            SELECT * FROM {table}
            WHERE Ticker = ?
            AND Date BETWEEN ? AND ?
            ORDER BY date;
            """
            df = pd.read_sql_query(query, conn, params=(ticker, start_date, end_date)) 

    elif table in ["Stocks", "Sector_1hot_encoded"]:
        if ticker.lower() == "all_stocks":
            query = f"""
            SELECT * FROM {table};
            """
            df = pd.read_sql_query(query, conn)
        else:
            query = f"""
            SELECT * FROM {table}
            WHERE Ticker = ?;
            """
            df = pd.read_sql_query(query, conn, params=[ticker])

    else:
        if ticker.lower() == "all_stocks":
            query = f"""
            SELECT * FROM {table}
            WHERE date BETWEEN ? AND ?
            ORDER BY date;
            """
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        else:
            query = f"""
            SELECT * FROM {table}
            WHERE Ticker = ?
            AND Date BETWEEN ? AND ?
            ORDER BY date;
            """
            df = pd.read_sql_query(query, conn, params=[ticker, start_date, end_date])

    # A lazy solution to avoid dynamically changing the data types in the database. 
    # If you read this, I am sorry to you have seen this barbaric method of database management.
    if table != "Stocks":
        for col in df.columns:
            if col not in ["Ticker", "Date", "FilingDate", "EffectiveDate", "Feature"]:
                df[col] = df[col].astype(float)
    
    conn.close()

    return df


def query_and_combine_TA_data(ticker= "NVDA", start_date = None, end_date = None):
    """
    This function should be used to query and combine technical analysis data from the database.
    The function should return a pandas DataFrame with the combined data for the given stock ticker and date range.
    """

    # Connect to the SQLite database
    conn = sqlite3.connect("../Data/EOD_financial_data.db")

    technicals_tables = ["AROON_10", "AROON_20", "Bollinger_Bands_10", "Bollinger_Bands_20", 
                         "EMA_10", "EMA_20", "MACD_10", "MACD_20",  "Max_High_10", "Max_High_20",
                            "Min_Low_10", "Min_Low_20", "RSI_10", "RSI_20", "ROC_10", "ROC_20"]

    # Create a SQL query string
    if start_date == None or end_date == None:
        start_date = "1996-01-01"
        end_date = "2024-10-31"

    combined_df = pd.DataFrame()
    all_dfs = []

    for idx,table in enumerate(technicals_tables):
        if ticker == "all_stocks":
            query = f"""
            SELECT * FROM {table}
            WHERE date BETWEEN ? AND ?
            ORDER BY date;
            """
            df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        else:
            query = f"""
            SELECT * FROM {table}
            WHERE Ticker = ?
            AND date BETWEEN ? AND ?
            ORDER BY date;
            """
            df = pd.read_sql_query(query, conn, params=[ticker, start_date, end_date])

        for col in df.columns:
            if col not in ["Ticker", "Date"]:
                df[col] = df[col].astype(float)

        all_dfs.append(df.copy(deep=True))
                
    conn.close()

    combined_df = all_dfs[0]
    for idx, df in enumerate(all_dfs):
        # merge the dataframes on Ticker and Date
        if idx == 0:
            continue
        combined_df = pd.merge(combined_df, df, on=["Date", "Ticker"], how="outer")

    combined_df = combined_df.sort_values(by=["Date"])
        
    return combined_df


def check_if_close_price_exists(conn, table_name, ticker):
    """
    Check if the close price for a given stock and date exists in the database.

    Parameters:
    cursor (sqlite3.Cursor): The SQLite database cursor.
    table_name (str): The name of the table to query.
    ticker (str): The stock ticker symbol to filter by.

    Returns:
    bool: True if the close price exists, False otherwise.
    """

    # Create a SQL query string
    query = f"""
    SELECT Close FROM {table_name}
    WHERE Ticker = ?
    LIMIT 1;
    """

    # Execute the query
    try:
        df = pd.read_sql_query(query, conn, params=[ticker])
        if df.empty:
            return False
        else: 
            return True
    except Exception:
        return False
    
    

def create_and_upload_table(df, table_name, column_names = ["Date", "Ticker", "Attribute1", "Attribute2"] , primary_key = ["Date", "Ticker"]):
    
    # Creates a new table in the database and uploads all the data from the DataFrame
    # if table doesn't exist, it is created

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor() 

    # Check if the Date is in the correct format YYYY-MM-DD
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df = df[df.columns.intersection(column_names)]

    columns_names_for_db = ', '.join([f'{column} TEXT' for column in column_names])
    for column in column_names:
        for blacklisted_column in ["Date", "Ticker"]:
            if blacklisted_column.lower() in column.lower():
                continue
 
        columns_names_for_db = columns_names_for_db.replace(f"{column} TEXT", f"{column} REAL")

    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_names_for_db}, PRIMARY KEY ({', '.join(primary_key)}));")

    # Insert the data into the table
    for index, row in df.iterrows(): 
        cursor.execute(f"INSERT OR IGNORE INTO {table_name} ({', '.join(column_names)}) VALUES ({', '.join(['?' for _ in column_names])});", tuple(row))

    cursor.connection.commit()


def fetch_all_trading_days(zero_pad = False, remove_first_3_months = False):
    """
    Fetch all trading days from the database.

    Returns:
    A list of trading days which contains data. 

    NOTE: The first 3 months are discarded:
    We remove the first 3 months of data to ensure that most stocks have data which is not NaN's replaced by 0'
    Most of the quarterly reports are also present after this period, as well as the technical indicators.abs
    Including the first 3 months of data would just add 3 months worth of noisy and incomplete data.
    A month worth of data includes roughly 21 trading days, considering holidays and weekends where the stock market is closed.
    """
    conn = sqlite3.connect("../Data/EOD_financial_data.db")

    query = """
    SELECT DISTINCT Date FROM Price_Data
    ORDER BY Date;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Format the date column as a list of strings with the format "YYYY-MM-DD"
    trade_dates = df["Date"].tolist()
    if remove_first_3_months:
        trade_dates = trade_dates[63:]

    if zero_pad:
        return trade_dates  
    else:
        return [date[:10] for date in trade_dates]


def fetch_all_tickers():
    """
    Fetch all tickers from the database.

    Returns:
    a list of strings with the tickers which have successfully been queried from the ML_Feature_Table.

    NOTE: This function can only be used after the ML_Feature_Table has been created. #LateGameStuff
    """

    conn = sqlite3.connect("../Data/EOD_financial_data.db")

    query = """
    SELECT DISTINCT Ticker FROM ML_Feature_Table
    ORDER BY Ticker;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df["Ticker"].tolist()


def zero_pad_date(date_list):
    """
    input assumes a list of strings with the format "YYYY-MM-DD"
    """
    return [date + " 00:00:00" for date in date_list]


def rmv_time(date_list):
    """
    input assumes a list of strings with the format "YYYY-MM-DD HH:MM:SS"
    """
    return [date[:10] for date in date_list]


def get_all_quarters(start_date = "1996-01-01", end_date = "2024-10-31"):
    # Return a list of all quarters between the start and end date
    quarters = []
    for year in range(int(start_date[:4]), int(end_date[:4]) + 1):
        for quarter in range(1, 13, 3):
            quarters.append(f"{year}-0{quarter}-01 00:00:00")

    return quarters


def get_percentage_change_for_backtester(start_date,end_date):

    # Setting an index on the Date column of Adjusted_adj_price_data if it doesnt already exist for future queries.

    pickle_path = f"../Data/backtester_pct_change__{start_date}__{end_date}.pkl"
    if os.path.exists(pickle_path):
        return pd.read_pickle(pickle_path)

    else:



        conn = sqlite3.connect("../Data/EOD_financial_data.db")
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(Adjusted_Price_Data);")
        columns = cursor.fetchall()
        columns = [column[1] for column in columns]
        if "Date" not in columns:
            cursor.execute("CREATE INDEX idx_date ON Adjusted_Price_Data(Date);")
            conn.commit()
        cursor.execute("PRAGMA table_info(Price_Data);")
        columns = cursor.fetchall()
        columns = [column[1] for column in columns]
        if "Date" not in columns:
            cursor.execute("CREATE INDEX idx_date ON Price_Data(Date);")
            conn.commit()

        adj_price_data = query_stock_data(table="Adjusted_Price_Data", ticker="all_stocks", start_date=start_date, end_date=end_date)

        pct_change_dict = {}
        #dictionary in shpe of {ticker: {date: percentage_change}}
        for idx, row in tqdm(adj_price_data.iterrows(), total=adj_price_data.shape[0], desc="Calculating returns for all stocks and dates in the dataset."):
            # Check if the date is a monday by converting it to datetime and checking the weekday
            ticker = row["Ticker"]
            date = row["Date"]
            if ticker not in pct_change_dict:
                pct_change_dict[ticker] = {}
            if pd.to_datetime(date).weekday() == 0:
                tmp_end_date = pd.to_datetime(date) + pd.DateOffset(hours = 1)
                price_data   = query_stock_data(table="Price_Data", ticker=ticker, start_date=date, end_date=tmp_end_date.strftime("%Y-%m-%d %H:%M:%S"))
                monday_open  = price_data["Open"].values[0]
                monday_close = price_data["Close"].values[0]
                pct_change = (monday_close - monday_open) / monday_open
            else:
                pct_change = row["Close"]
            pct_change_dict[ticker][row["Date"]] = pct_change

        conn.close()
        with open(pickle_path, "wb") as f:
            pickle.dump(pct_change_dict, f)
        
        return pct_change_dict


if __name__ == "__main__":
    # Test the query_stock_data function
    # stock_data = query_stock_data(table="Price_Data", ticker="NVDA", start_date="1996-01-01", end_date="2024-10-31")
    # print(stock_data.head())
    # print(stock_data.tail())

    # print("")
    # stock_data = query_stock_data(table="Balance_Sheets", ticker="NVDA", start_date="1996-01-01", end_date="2024-10-31")
    # print(stock_data.head())
    # print(stock_data.tail())

    # print("")
    # stock_data = query_stock_data(table="Balance_Sheets", ticker="NVDA", start_date="1996-01-01", end_date="2024-10-31")
    # print(stock_data.head())
    # print(stock_data.tail())

    # print("")
    # stock_data = query_stock_data(table="Balance_Sheets", ticker="all_stocks", start_date="1996-01-01", end_date="2024-10-31")
    # print(stock_data.head())
    # print(stock_data.tail())

    # print("")
    # stock_data = query_stock_data(table="Stocks")
    # print(stock_data.head())
    # print(stock_data.tail())

    # Test the query_and_combine_TA_data())
    print("Testing query_and_combine_TA_data")
    print("")
    combined_TA_df = query_and_combine_TA_data(ticker= "NVDA", start_date = None, end_date = None)
    print(combined_TA_df.head())
    print(combined_TA_df.tail())

    # Test query_stock_data for MACD_10
    # stock_data = query_stock_data(table="MACD_10", ticker="all_stocks")
    # print(stock_data.head())
    # print(stock_data.tail())
    # print("")
    # print(stock_data.describe())