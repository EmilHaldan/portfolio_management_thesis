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



def get_credentials(target_name):
    creds = win32cred.CredRead(target_name, win32cred.CRED_TYPE_GENERIC)
    credential_blob = creds["CredentialBlob"]
    password = credential_blob.decode('utf-16-le').strip()
    return password


def create_expected_reports(all_tickers):
    if os.path.exists("../Data/expected_reports.pkl"):
        with open("../Data/expected_reports.pkl", "rb") as f:
            expected_reports = pickle.load(f)
        # for key,value in expected_reports.items():
        #     print(key, ":", value)
        return
    expected_reports = {ticker: {"Balance_Sheets": 0, "Income_Statements": 0, "Cash_Flow": 0, "Price_Data": 0, "Earnings": 0, "Ticker_Not_Found_From_EOD": False} for ticker in all_tickers}
    try:
        with open("../Data/expected_reports.pkl", "wb") as f:
            pickle.dump(expected_reports, f)
    except PermissionError:
        time.sleep(1)
        with open("../Data/expected_reports.pkl", "wb") as f:
            pickle.dump(expected_reports, f)


def lookup_and_expected_reports(ticker):
    # Lookup the expected reports for a given stock ticker
    altered = False
    with open("../Data/expected_reports.pkl", "rb") as f:
        expected_reports = pickle.load(f)
        if ticker not in expected_reports:
            altered = True
            expected_reports[ticker] = {"Balance_Sheets": 0, "Income_Statements": 0, "Cash_Flow": 0, "Price_Data": 0, "Earnings": 0, "Ticker_Not_Found_From_EOD": False}
    if altered:
        with open("../Data/expected_reports.pkl", "wb") as f:
            pickle.dump(expected_reports, f)
    return expected_reports[ticker]


def update_expected_reports(ticker, table_name, table_length):
    # Update the expected reports for a given stock ticker
    with open("../Data/expected_reports.pkl", "rb") as f:
        expected_reports = pickle.load(f)
        expected_reports[ticker][table_name] = table_length
    with open("../Data/expected_reports.pkl", "wb") as f:
        pickle.dump(expected_reports, f)


def updated_expected_reports_ticker_not_found(ticker):
    with open("../Data/expected_reports.pkl", "rb") as f:
        expected_reports = pickle.load(f)
        expected_reports[ticker]["Ticker_Not_Found_From_EOD"] = True
    with open("../Data/expected_reports.pkl", "wb") as f:
        pickle.dump(expected_reports, f)


def get_fundamentals_data_safe(api_client, ticker):
    try:
        # Try fetching the data from the API
        data = api_client.get_fundamentals_data(ticker)
        return data
    
    except requests.exceptions.JSONDecodeError as e:
        # Handle the case where the response is not JSON or is malformed
        print(f"JSONDecodeError: Failed to fetch data for ticker: {ticker}. Error: {e}")
        
        # Optionally, log the error for further analysis
        # You can log this to a file or a monitoring system
        with open('error_log.txt', 'a') as f:
            f.write(f"JSONDecodeError for {ticker}: {str(e)}\n")
        
        # Return None or some default value to handle the error gracefully in your code
        return None
    
    except requests.exceptions.RequestException as e:
        # Catch other network-related errors (timeouts, connection errors, etc.)
        print(f"RequestException: Failed to fetch data for ticker: {ticker}. Error: {e}")
        with open('error_log.txt', 'a') as f:
            f.write(f"RequestException for {ticker}: {str(e)}\n")
        return None
    
    except Exception as e:
        # Catch all other exceptions
        print(f"Unexpected error occurred for ticker {ticker}: {e}")
        with open('error_log.txt', 'a') as f:
            f.write(f"Unexpected error for {ticker}: {str(e)}\n")
        return None


def log_data_population(ticker, table_name, action, ticker_data_fetch_timer = 0):
    cur_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_to_complete = time.time() - ticker_data_fetch_timer
    with open('logs/data_pop_log.txt', 'a') as f:
        if action == "Avoiding":
            f.write(f"TIMESTAMP: {cur_date_time} TABLENAME: {table_name}  TICKER: {ticker} Avoiding populating database \n")
        elif action == "Populating":
            f.write(f"TIMESTAMP: {cur_date_time} TABLENAME: {table_name}  TICKER: {ticker} Populating database \n")
        elif action == "Successful":
            f.write(f"TIMESTAMP: {cur_date_time} TABLENAME: {table_name}  TICKER: {ticker} Successfully Populated database \n")
        elif action == "Complete":
            f.write(f"TIMESTAMP: {cur_date_time} TICKER: {ticker} Completed Database Population in {time_to_complete:.2f} seconds \n")
            f.write(f"\n\n")
        elif action == "Starting Script":
            f.write("\n\n")
            f.write(f"TIMESTAMP: {cur_date_time} STARTING DATABASE POPULATION SCRIPT \n")
            f.write(f"------------------------------------------------------------\n\n")
        else:
            f.write(f"TIMESTAMP: {cur_date_time} TABLENAME: {table_name}  TICKER: {ticker}  ERROR MESSAGE: {action} \n")


def log_data_validation(ticker, table_name, action):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    cur_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('logs/data_pop_log.txt', 'a') as f:
        f.write(f"TIMESTAMP: {cur_date_time} TABLENAME: {table_name}  TICKER: {ticker} {action} \n")


def log_missing_data(ticker, message):
    cur_date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('logs/missing_data_log.txt', 'a') as f:
        f.write(f"TIMESTAMP: {cur_date_time} TICKER: {ticker} DATA: {message} \n")


def populate_missing_general_data(eod_financials_resp, ticker):
    critical_data_missing = False
    if eod_financials_resp is None:
        log_missing_data(ticker, f"Nothing returned from API")
        critical_data_missing = True
        return eod_financials_resp, critical_data_missing

    if "General" not in eod_financials_resp:
        log_missing_data(ticker, "General")
        critical_data_missing = True
        return eod_financials_resp, critical_data_missing

    else:
         
        if "Financials" not in eod_financials_resp:
            log_missing_data(ticker, "Financials")
            critical_data_missing = True
            return eod_financials_resp, critical_data_missing
            
        else:
            if "Balance_Sheet" not in eod_financials_resp['Financials']:
                log_missing_data(ticker, "Financials/Balance_Sheet")
                critical_data_missing = True
                return eod_financials_resp, critical_data_missing

            if "quarterly" not in eod_financials_resp['Financials']['Balance_Sheet']:
                log_missing_data(ticker, "Financials/Balance_Sheet/quarterly")
                critical_data_missing = True
                return eod_financials_resp, critical_data_missing

            if "Income_Statement" not in eod_financials_resp['Financials']:
                log_missing_data(ticker, "Financials/Income_Statement")
                critical_data_missing = True
                return eod_financials_resp, critical_data_missing

            stock_info = eod_financials_resp['General']
            if "Name" not in stock_info:
                eod_financials_resp['General']['Name'] = "N/A"
                log_missing_data(ticker, "General/Name")
            if "CurrencyCode" not in stock_info:
                eod_financials_resp['General']['CurrencyCode'] = "N/A"
                log_missing_data(ticker, "General/CurrencyCode")
            if "Sector" not in stock_info:
                eod_financials_resp['General']['Sector'] = "N/A"
                log_missing_data(ticker, "General/Sector")
            if "Industry" not in stock_info:
                eod_financials_resp['General']['Industry'] = "N/A"
                log_missing_data(ticker, "General/Industry")
            if "IPODate" not in stock_info:
                eod_financials_resp['General']['IPODate'] = "N/A"
                log_missing_data(ticker, "General/IPODate")
            if "FiscalYearEnd" not in stock_info:
                eod_financials_resp['General']['FiscalYearEnd'] = "N/A"
                log_missing_data(ticker, "General/FiscalYearEnd")

    return eod_financials_resp, critical_data_missing


def populate_stocks_table(eod_financials_resp, sql_conn, sql_cursor, ticker):

    stocks_table_pressence = sql_cursor.execute("SELECT * FROM Stocks WHERE Ticker = ?", (ticker,)).fetchone()
    
    if stocks_table_pressence is None:

        log_data_population(ticker, "Stocks", "Populating")
        stock_info = eod_financials_resp['General']

        sql_cursor.execute("INSERT INTO Stocks (Ticker, Name, Currency, Sector, Industry, IPODate, FiscalYearEnd) VALUES (?, ?, ?, ?, ?, ?, ?)", 
            (ticker, stock_info['Name'], stock_info['CurrencyCode'], stock_info['GicSector'], 
            stock_info['Industry'], stock_info['IPODate'], stock_info['FiscalYearEnd']))

        sql_conn.commit()
        log_data_population(ticker, "Stocks", "Successful")

    else:
        log_data_population(ticker, "Stocks", "Avoiding")



def populate_balance_sheet_table(eod_financials_resp, sql_conn, sql_cursor, ticker,  start_date, end_date):

    balance_sheet_table_pressence = sql_cursor.execute("SELECT * FROM Balance_Sheets WHERE Ticker = ?", (ticker,)).fetchall()

    if len(balance_sheet_table_pressence) < len(eod_financials_resp["Financials"]["Balance_Sheet"]["quarterly"]):
        log_data_population(ticker, "Balance_Sheets", "Populating")
        quarterly_statement_dates = eod_financials_resp["Financials"]["Balance_Sheet"]["quarterly"].keys()
        dates_ignored = []

        for date in quarterly_statement_dates:
            if date < start_date or date > end_date:
                dates_ignored.append(date)
                continue
            if date in balance_sheet_table_pressence:
                continue

            balance_sheet = eod_financials_resp["Financials"]["Balance_Sheet"]["quarterly"][date]
            balance_sheet['filing_date'] = add_1_day_to_filing_date(balance_sheet['filing_date'], balance_sheet['date'])

            sql_cursor.execute("""INSERT OR IGNORE INTO Balance_Sheets (Ticker, FilingDate, EffectiveDate, TotalAssets, TotalCurrentAssets, IntangibleAssets, 
                                                        OtherCurrentAssets, TotalLiabilities, OtherCurrentLiab, NetDebt, ShortTermDebt, 
                                                        LongTermDebt, TotalStockholderEquity, RetainedEarnings, GoodWill, Cash, Inventory,
                                                        OutstandingShares) 
                                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                                        (ticker, balance_sheet['filing_date'], balance_sheet['date'],
                                                        balance_sheet['totalAssets'], balance_sheet['totalCurrentAssets'],  balance_sheet['intangibleAssets'],
                                                        balance_sheet['otherCurrentAssets'], balance_sheet['totalLiab'], balance_sheet['otherCurrentLiab'],
                                                        balance_sheet['netDebt'], balance_sheet['shortTermDebt'], balance_sheet['longTermDebt'],
                                                        balance_sheet['totalStockholderEquity'], balance_sheet['retainedEarnings'], balance_sheet['goodWill'],
                                                        balance_sheet['cash'], balance_sheet['inventory'], balance_sheet['commonStockSharesOutstanding']))

        update_expected_reports(ticker, "Balance_Sheets", len(quarterly_statement_dates) - len(dates_ignored))

        sql_conn.commit()
        log_data_population(ticker, "Balance_Sheets", "Successful")

    else:
        log_data_population(ticker, "Balance_Sheets", "Avoiding")



def populate_income_statement_table(eod_financials_resp, sql_conn, sql_cursor, ticker, start_date, end_date):

    income_statement_table_pressence = sql_cursor.execute("SELECT * FROM Income_Statements WHERE Ticker = ?", (ticker,)).fetchall()

    if len(income_statement_table_pressence) < len(eod_financials_resp["Financials"]["Income_Statement"]["quarterly"]):
        log_data_population(ticker, "Income_Statements", "Populating")
        income_statement_dates = eod_financials_resp["Financials"]["Income_Statement"]["quarterly"].keys()
        dates_ignored = []

        for date in income_statement_dates:
            if date < start_date or date > end_date:
                dates_ignored.append(date)
                continue
            if date in income_statement_table_pressence:
                continue

            income_statement = eod_financials_resp["Financials"]["Income_Statement"]["quarterly"][date]
            income_statement['filing_date'] = add_1_day_to_filing_date(income_statement['filing_date'], income_statement['date'])

            sql_cursor.execute("""INSERT OR IGNORE INTO Income_Statements (Ticker, FilingDate, EffectiveDate, ResearchDevelopment, IncomeBeforeTax, NetIncome, 
                                                        SellingGeneralAdministrative, GrossProfit, EBIT, EBITDA, OperatingIncome, OtherOperatingExpenses, 
                                                        InterestExpense, TaxProvision, InterestIncome, NetInterestIncome, IncomeTaxExpense, TotalRevenue, 
                                                        totalOperatingExpenses, costOfRevenue, totalOtherIncomeExpenseNet, netIncomeFromContinuingOps) 
                                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,?, ?, ?, ?)""",
                                                        (ticker, income_statement['filing_date'], income_statement['date'],
                                                        income_statement['researchDevelopment'], income_statement['incomeBeforeTax'], income_statement['netIncome'],
                                                        income_statement['sellingGeneralAdministrative'], income_statement['grossProfit'], income_statement['ebit'],
                                                        income_statement['ebitda'], income_statement['operatingIncome'], income_statement['otherOperatingExpenses'],
                                                        income_statement['interestExpense'], income_statement['taxProvision'], income_statement['interestIncome'],
                                                        income_statement['netInterestIncome'], income_statement['incomeTaxExpense'], income_statement['totalRevenue'],
                                                        income_statement['totalOperatingExpenses'], income_statement['costOfRevenue'], income_statement['totalOtherIncomeExpenseNet'],
                                                        income_statement['netIncomeFromContinuingOps']))

        update_expected_reports(ticker, "Income_Statements", len(income_statement_dates) - len(dates_ignored))

        sql_conn.commit()
        log_data_population(ticker, "Income_Statements", "Successful")

    else: 
        log_data_population(ticker, "Income_Statements", "Avoiding")


def populate_earnings_table(eod_financials_resp, sql_conn, sql_cursor, ticker, start_date, end_date):

    earnings_sheet_table_pressence = sql_cursor.execute("SELECT * FROM Earnings WHERE Ticker = ?", (ticker,)).fetchall()

    if len(earnings_sheet_table_pressence) < len(eod_financials_resp["Earnings"]["History"]):
        log_data_population(ticker, "Earnings", "Populating")
        earnings_dates = eod_financials_resp["Earnings"]["History"].keys()
        dates_ignored = []

        for date in earnings_dates:
            if date < start_date or date > end_date:
                dates_ignored.append(date)
                continue
            if date in earnings_sheet_table_pressence:
                continue

            earnings = eod_financials_resp["Earnings"]["History"][date]
            earnings['reportDate'] = add_1_day_to_filing_date(earnings['reportDate'], earnings['date'])

            sql_cursor.execute("""INSERT OR IGNORE INTO Earnings (Ticker, FilingDate, EffectiveDate, EPSActual, EPSEstimate, EPSDifference, SurprisePercent)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (ticker, earnings['reportDate'], earnings['date'], earnings['epsActual'], earnings['epsEstimate'], earnings['epsDifference'], earnings['surprisePercent']))

        update_expected_reports(ticker, "Earnings", len(earnings_dates) - len(dates_ignored))
        sql_conn.commit()
        log_data_population(ticker, "Earnings", "Successful")

    else:
        log_data_population(ticker, "Earnings", "Avoiding")



def populate_cash_flow_table(eod_financials_resp, sql_conn, sql_cursor, ticker,  start_date, end_date):

    cash_flow_table_pressence = sql_cursor.execute("SELECT * FROM Cash_Flow WHERE Ticker = ?", (ticker,)).fetchall()

    if len(cash_flow_table_pressence) < len(eod_financials_resp["Financials"]["Cash_Flow"]["quarterly"]):
        log_data_population(ticker, "Cash_Flow", "Populating")

        dates_ignored = []
        cash_flow_dates = eod_financials_resp["Financials"]["Cash_Flow"]["quarterly"].keys()
        for date in cash_flow_dates:
            if date < start_date or date > end_date:
                dates_ignored.append(date)
                continue
            if date in cash_flow_table_pressence:
                continue

            cash_flow = eod_financials_resp["Financials"]["Cash_Flow"]["quarterly"][date]
            cash_flow['filing_date'] = add_1_day_to_filing_date(cash_flow['filing_date'], cash_flow['date'])

            sql_cursor.execute("""INSERT OR IGNORE INTO Cash_Flow (Ticker, FilingDate, EffectiveDate, Investments, NetIncome, BeginPeriodCashFlow,
                                                    EndPeriodCashFlow, ChangeInCash, ChangeToInventory, FreeCashFlow, CapitalExpenditures) 
                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", 
                                                    (ticker, cash_flow['filing_date'], cash_flow['date'],
                                                    cash_flow['investments'], cash_flow['netIncome'], cash_flow['beginPeriodCashFlow'],
                                                    cash_flow['endPeriodCashFlow'], cash_flow['changeInCash'], cash_flow['changeToInventory'],
                                                    cash_flow['freeCashFlow'], cash_flow['capitalExpenditures']))
                                                
        update_expected_reports(ticker, "Cash_Flow", len(cash_flow_dates) - len(dates_ignored))

        sql_conn.commit()
        log_data_population(ticker, "Cash_Flow", "Successful")

    else:
        log_data_population(ticker, "Cash_Flow", "Avoiding")


def validate_price_data(candle_prices, ticker):
    """
    In some instances, some of the price data had entries where the stock would drop by 99% in a single day, and then go back up the next day.
    This is obviously a mistake in the data, and we need to interpolate the data to fix this issue.
    This is done by using the previous day's prices.

    Instances such as GameStop (GME), can be used as an upper limit for acceptable price changes.

    If a stock increases decreases by 90% in a single day, then the data is likely incorrect.
    """
    
    local_candle_prices = candle_prices.copy()
    local_candle_prices["pct_change"] = local_candle_prices["close"].pct_change()
    smallest_pct_change = local_candle_prices["pct_change"].min()
    biggest_pct_change = local_candle_prices["pct_change"].max()

    if (smallest_pct_change < -0.9):
        print(f"The stock {ticker} will not be added to the database due to inconsistent price data")
        print("Biggest Percentage Change: ", biggest_pct_change)
        print("Smallest Percentage Change: ", smallest_pct_change)
        print("")
        return False
    else:
        return True


def populate_price_data_table(api_client, sql_conn, sql_cursor, ticker, start_date, end_date):

    price_data_table_pressence = sql_cursor.execute("SELECT * FROM Price_Data WHERE Ticker = ?", (ticker,)).fetchall()

    resp = api_client.get_eod_historical_stock_market_data(symbol = ticker, period='d', from_date = start_date, to_date = end_date, order='a')
    candle_prices = pd.DataFrame(resp)

    if candle_prices.empty:
        log_data_population(ticker, "Price_Data", "No price data available for the given date range")
        log_data_population(ticker, "Price_Data", "Avoiding")
        updated_expected_reports_ticker_not_found(ticker)
        return

    log_data_population(ticker, "Price_Data", "Populating")
    candle_prices["date"] = pd.to_datetime(candle_prices["date"])
    candle_prices = candle_prices[(candle_prices["date"] >= start_date) & (candle_prices["date"] <= end_date)]

    candle_prices["adjusted_scalar"] = candle_prices["adjusted_close"] / candle_prices["close"]
    candle_prices["open"] = candle_prices["open"] * candle_prices["adjusted_scalar"]
    candle_prices["high"] = candle_prices["high"] * candle_prices["adjusted_scalar"]
    candle_prices["low"] = candle_prices["low"] * candle_prices["adjusted_scalar"]
    candle_prices["close"] = candle_prices["close"] * candle_prices["adjusted_scalar"]
    candle_prices["volume"] = candle_prices["volume"] * candle_prices["adjusted_scalar"]
    candle_prices["ticker"] = ticker

    # Validating price data before inserting into the database
    if validate_price_data(candle_prices, ticker):
        # change order of columns to match the database
        candle_prices = candle_prices[['ticker', 'date', 'open', 'high', 'low', 'close', 'adjusted_close', 'adjusted_scalar', 'volume']]
        candle_prices.columns = [col.capitalize() for col in candle_prices.columns]
        candle_prices.to_sql('Price_Data', sql_conn, if_exists='append', index=False)

        update_expected_reports(ticker, "Price_Data", len(candle_prices))

        sql_conn.commit()
        log_data_population(ticker, "Price_Data", "Successful")
        return True
    else: 
        log_data_population(ticker, "Price_Data", "No price data available for the given date range")
        return False



def validate_ticker_in_db(ticker, sql_conn, sql_cursor, start_date, end_date):

    stocks_table_pressence = sql_cursor.execute("SELECT * FROM Stocks WHERE Ticker = ?", (ticker,)).fetchone()
    balance_sheet_table_pressence = sql_cursor.execute("SELECT * FROM Balance_Sheets WHERE Ticker = ?", (ticker,)).fetchall()
    income_statement_table_pressence = sql_cursor.execute("SELECT * FROM Income_Statements WHERE Ticker = ?", (ticker,)).fetchall()
    cash_flow_table_pressence = sql_cursor.execute("SELECT * FROM Cash_Flow WHERE Ticker = ?", (ticker,)).fetchall()
    price_data_table_pressence = sql_cursor.execute("SELECT * FROM Price_Data WHERE Ticker = ?", (ticker,)).fetchall()

    data_download_required = {"Stocks": False, "Balance_Sheets": False, "Income_Statements": False, "Cash_Flow": False, 
                                "Price_Data": False, "Ticker_Not_Found_From_EOD": False, "Earnings": False}

    expected_reports = lookup_and_expected_reports(ticker)

    if expected_reports["Ticker_Not_Found_From_EOD"]:
        data_download_required["Ticker_Not_Found_From_EOD"] = True
        return data_download_required

    if stocks_table_pressence is None:
        log_data_validation(ticker, "Stocks", "Ticker not found in the database, will populate the database with data for {}".format(ticker))
        data_download_required["Stocks"] = True

    if (expected_reports["Earnings"] == 0) or len(balance_sheet_table_pressence) < expected_reports["Earnings"] -1:
        log_data_validation(ticker, "Earnings", "Insufficient data in the database, will populate the database with data for {}".format(ticker))
        data_download_required["Earnings"] = True

    if (expected_reports["Balance_Sheets"] == 0) or len(balance_sheet_table_pressence) < expected_reports["Balance_Sheets"] -1:
        log_data_validation(ticker, "Balance_Sheets", "Insufficient data in the database, will populate the database with data for {}".format(ticker))
        data_download_required["Balance_Sheets"] = True

    if (expected_reports["Income_Statements"] == 0) or len(income_statement_table_pressence) < expected_reports["Income_Statements"] -1:
        log_data_validation(ticker, "Income_Statements", "Insufficient data in the database, will populate the database with data for {}".format(ticker))
        data_download_required["Income_Statements"] = True

    if (expected_reports["Cash_Flow"] == 0) or len(cash_flow_table_pressence) < expected_reports["Cash_Flow"] -1:
        log_data_validation(ticker, "Cash_Flow", "Insufficient data in the database, will populate the database with data for {}".format(ticker))
        data_download_required["Cash_Flow"] = True

    if (expected_reports["Price_Data"] == 0) or len(price_data_table_pressence) < expected_reports["Price_Data"] -1:
        log_data_validation(ticker, "Price_Data", "Insufficient data in the database, will populate the database with data for {}".format(ticker))
        data_download_required["Price_Data"] = True


    if sum(data_download_required.values()) == 0:
        log_data_validation(ticker, "ALL TABLES", " - ALL TABLES have sufficient data for {}".format(ticker))

    return data_download_required


def populate_database(start_date = "1996-01-01", end_date = "2024-07-01"):
    log_data_population("", "", "Starting Script")

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()

    creds = get_credentials("eodhd.com")
    api_client = APIClient(creds)

    spy_df, stock_tickers =  load_SPY_components()

    # NOTE: THIS LINE IS TEMPORARY WHILE DEVELOPING THE PIPELINE, AND SHOULD BE REPLACED
    stock_tickers = ["NVDA", "AMZN", "MSFT","AAL", "AAPL", "BAC","META", "GME"]  
    # stock_tickers = stock_tickers[:200] + ["NVDA", "AMZN", "MSFT","NSI", "AAPL", "VAT" ,"BAC","META", "GME", "CIN", "MNK"]  
    # stock_tickers = list(set(stock_tickers))
    # 200 Random stock, and some additional stocks with properties good for debugging and testing functionality

    create_expected_reports(stock_tickers)

    for ticker in tqdm(stock_tickers):
        ticker_data_fetch_timer = time.time()
        data_download_required = validate_ticker_in_db(ticker, conn, cursor, start_date, end_date)

        if data_download_required["Ticker_Not_Found_From_EOD"]:
            log_data_population(ticker, "ALL TABLES", "{} Not Found From EOD \n \n".format(ticker))
            continue

        if sum(data_download_required.values()) > 0:
            eod_financials_resp = get_fundamentals_data_safe(api_client, ticker = ticker)
            if eod_financials_resp is None:
                continue
        
            eod_financials_resp, critical_data_missing = populate_missing_general_data(eod_financials_resp, ticker)

            if not critical_data_missing:
                if data_download_required["Price_Data"]:
                    price_good = populate_price_data_table(api_client, conn, cursor, ticker, start_date, end_date)
                if price_good:
                    if data_download_required["Stocks"]:
                        populate_stocks_table(eod_financials_resp, conn, cursor, ticker)
                    if data_download_required["Earnings"]:
                        populate_earnings_table(eod_financials_resp, conn, cursor, ticker, start_date, end_date)
                    if data_download_required["Balance_Sheets"]:
                        populate_balance_sheet_table(eod_financials_resp, conn, cursor, ticker, start_date, end_date)
                    if data_download_required["Income_Statements"]:
                        populate_income_statement_table(eod_financials_resp, conn, cursor, ticker, start_date, end_date)
                    if data_download_required["Cash_Flow"]:
                        populate_cash_flow_table(eod_financials_resp, conn, cursor, ticker, start_date, end_date)
                
            else: 
                log_data_population(ticker, "ALL TABLES", "Critical data missing for {}".format(ticker))
                updated_expected_reports_ticker_not_found(ticker)

        log_data_population(ticker, "ALL TABLES", "Complete", ticker_data_fetch_timer)

    # Adding the SPY ETF to the database for benchmarking purposes
    data_download_required = validate_ticker_in_db("SPY", conn, cursor, start_date, end_date)
    price_good = populate_price_data_table(api_client, conn, cursor, "SPY", start_date, end_date)

    conn.close()


def add_1_day_to_filing_date(date: str, effective_date :str):
    """ 
    Converts date from YYYY-MM-DD to a datetime object, addd 1 day to it, and returns it back to string format.
    There are some instances of missing values for the filing date, so we create a disadavantagious transformation
        by making the filing date 2 months after the effective date, ensuring that there is no information leakage.
        This scenario is rare, so it won't have a significant impact on the dataset as a whole.
    There are also a few instances where the filing date is before the effective date, so we let the filing date 2 months + effective date.
    """
    if date == None:
        new_date = datetime.strptime(effective_date, "%Y-%m-%d") + pd.DateOffset(months=2)
    else:
        if date < effective_date:
            new_date = datetime.strptime(effective_date, "%Y-%m-%d") + pd.DateOffset(months=2)
        else:
            new_date = datetime.strptime(date, "%Y-%m-%d") + pd.DateOffset(days=1)
    return new_date.strftime("%Y-%m-%d")



def create_database():
    # Creating a new database using SQLite

    conn = sqlite3.connect("../Data/EOD_financial_data.db")
    cursor = conn.cursor()


    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Stocks (
            Ticker TEXT PRIMARY KEY,
            Name TEXT,
            Currency TEXT,
            Sector TEXT,
            Industry TEXT,
            IPODate TEXT,
            FiscalYearEnd TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Earnings (
            Ticker TEXT,
            FilingDate TEXT,
            EffectiveDate TEXT,
            EPSActual REAL,
            EPSEstimate REAL,
            EPSDifference REAL,
            SurprisePercent REAL,
            PRIMARY KEY (Ticker, FilingDate)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Balance_Sheets (
            Ticker TEXT,
            FilingDate TEXT,
            EffectiveDate TEXT,
            TotalAssets REAL,
            TotalCurrentAssets REAL,
            IntangibleAssets REAL,
            OtherCurrentAssets REAL,
            TotalLiabilities REAL,
            OtherCurrentLiab REAL,
            NetDebt REAL,
            ShortTermDebt REAL,
            LongTermDebt REAL,
            TotalStockholderEquity REAL,
            RetainedEarnings REAL,
            GoodWill REAL,
            Cash REAL,
            Inventory REAL,
            OutstandingShares REAL,
            PRIMARY KEY (Ticker, FilingDate)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Cash_Flow (
            Ticker TEXT,
            FilingDate TEXT,
            EffectiveDate TEXT,
            Investments REAL,
            NetIncome REAL,
            BeginPeriodCashFlow REAL,
            EndPeriodCashFlow REAL,
            ChangeInCash REAL,
            ChangeToInventory REAL,
            FreeCashFlow REAL,
            CapitalExpenditures REAL,
            PRIMARY KEY (Ticker, FilingDate)
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Income_Statements (
            Ticker TEXT,
            FilingDate TEXT,
            EffectiveDate TEXT,
            ResearchDevelopment REAL,
            IncomeBeforeTax REAL,
            NetIncome REAL,
            SellingGeneralAdministrative REAL,
            GrossProfit REAL,
            EBIT REAL,
            EBITDA REAL,
            OperatingIncome REAL,
            OtherOperatingExpenses REAL,
            InterestExpense REAL,
            TaxProvision REAL,
            InterestIncome REAL,
            NetInterestIncome REAL,
            IncomeTaxExpense REAL,
            TotalRevenue REAL,
            totalOperatingExpenses REAL,
            costOfRevenue REAL,
            totalOtherIncomeExpenseNet REAL,
            netIncomeFromContinuingOps REAL,
            PRIMARY KEY (Ticker, FilingDate)
        )
    ''')


    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Price_Data (
            Ticker TEXT,
            Date TEXT,
            Open REAL,
            High REAL,
            Low REAL,
            Close REAL,
            Adjusted_Close REAL,
            Adjusted_Scalar REAL,
            Volume REAL,
            PRIMARY KEY (Ticker, Date)
        )
    ''')


    conn.commit()
    conn.close()
    

def create_directories():
    """
    Creates paths for storing data.
    """
    os.makedirs("../Data/Nodes", exist_ok=True)
    os.makedirs("../Data/Edges", exist_ok=True)
    os.makedirs("../Data/Networks", exist_ok=True)
    os.makedirs("../Data/Networks_chunks", exist_ok=True)


if __name__ == "__main__":


    create_database()
    create_directories()
    # populate_database(start_date="1996-01-01", end_date="2024-07-01")

    # Used for testing purposes to populate the database with a smaller date range
    # populate_database(start_date="2014-01-01", end_date="2024-07-01")
    populate_database(start_date="2023-01-01", end_date="2024-07-01")
    


    



