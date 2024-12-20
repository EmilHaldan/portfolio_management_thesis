import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from tqdm import tqdm
import pickle
import time 

from datetime import datetime
from datetime import timedelta

from create_financial_database import get_credentials 
from SQLite_tools import query_stock_data, check_if_close_price_exists, get_percentage_change_for_backtester
from ticker_loader import load_SPY_components



def load_results(filename):
    """
    Load the results from the database
    """
    with open(f'../Data/Results/{filename}.pkl', 'rb') as f:
        results = pickle.load(f)
    print("Dict keys: ", results.keys())
    print("Learning_Rate", results['Learning_Rate'])
    return results



def load_index_to_ticker_dict():
    """
    Load the index to ticker dictionary
    """
    with open('../Data/Nodes/index_to_ticker.pkl', 'rb') as f:
        index_to_ticker = pickle.load(f)

    with open('../Data/Nodes/ticker_to_index.pkl', 'rb') as f:
        ticker_to_index = pickle.load(f)

    return index_to_ticker, ticker_to_index



def get_SPY_OHLC_data():
    """
    Get the Open, High, Low, Close, Volume data of the SPY ETF from 1996 to 2024
    """
    SPY_data = query_stock_data(ticker = "SPY", table = "Adjusted_Price_Data")
    return SPY_data


def long_strategy_1(predictions, current_portfolio, available_cash, min_total_weight_scalar, k_stocks = 15):
    """
    Strategy: Allocate weights based on predictions exceeding a threshold.

    Parameters:
        predictions (dict): Dictionary of {ticker: prediction}.
        current_portfolio (dict): Current portfolio composition.
        available_cash (float): Available cash to invest.
        threshold (float): Minimum prediction value to consider a stock.

    Returns:
        dict: New portfolio composition including cash weight.
    """
    ticker_to_use = k_stocks
    max_weight_pr_stock = 4/ticker_to_use
    min_total_weight =  ticker_to_use*min_total_weight_scalar    
    
    top_tickers = sorted(predictions, key=predictions.get, reverse=True)[:ticker_to_use]
    # weights = {ticker: predictions[ticker] for ticker in top_tickers if predictions[ticker] > threshold}
    weights = {ticker: predictions[ticker] for ticker in top_tickers}

    # Sorting the weights dict by the ticker name for easier inspection
    weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[0])}

    if len(weights) == 0:
        # hold all in cash
        return {"cash": 1.0}
    else:
        # Calculate weights based on prediction
        total_weight = sum(weights.values())
        if total_weight < min_total_weight:
            # If the total weight is less than the minimum, we use the minimum weight
            # To ensure that if the model predicts the stocks for the next week to be "shit", 
            # it will keep most of its worth in cash instead of stocks.
            total_weight = min_total_weight
        new_portfolio = {ticker: (weight / total_weight) for ticker, weight in weights.items()}
        for ticker, weight in new_portfolio.items():
            if weight > max_weight_pr_stock:
                new_portfolio[ticker] = max_weight_pr_stock
        new_portfolio["cash"] = 1 - sum(new_portfolio.values())
        if new_portfolio["cash"] < 0.00001:
            new_portfolio["cash"] = 0

    return new_portfolio


def group_dates_by_month(dates):
    """
    Group dates by the end of each month.

    Parameters:
        dates (pd.Series): Series of dates.

    Returns:
        list: List of lists of dates grouped by the end of the month.
    """
    return dates.to_series().groupby(dates.to_series().dt.to_period("M")).apply(list).to_list()


def get_predictions_for_prior_two_weeks(month_start, all_dates, all_predictions, index_to_ticker):
    """
    Retrieve predictions for the two weeks prior to the start of the month.

    Parameters:
        month_start (str): Start date of the month in 'YYYY-MM-DD' format.
        all_dates (pd.Series): All dates from the results dictionary.
        all_predictions (list): List of prediction arrays corresponding to all dates.
        index_to_ticker (dict): Mapping of indices to stock tickers.

    Returns:
        dict: Dictionary mapping tickers to their averaged predictions for the two weeks prior.
    """
    month_start_date = pd.to_datetime(month_start)
    prior_two_weeks_start = month_start_date - timedelta(weeks=2)

    relevant_dates = all_dates[(all_dates >= prior_two_weeks_start) & (all_dates < month_start_date)]
    predictions = {}

    for idx,date in enumerate(relevant_dates):
        pred_idx = all_dates.get_loc(date)
        prediction_values = all_predictions[pred_idx]
        for node_idx, pred in enumerate(prediction_values):
            ticker = index_to_ticker[node_idx]
            if ticker not in predictions:
                predictions[ticker] = []
            predictions[ticker].append(pred*(idx/10))

    averaged_predictions = {ticker: np.mean(preds) for ticker, preds in predictions.items()}
    return averaged_predictions


def calculate_daily_percentage_increase(state_dict, month_dates, pct_change_dict, trading_cost, stop_loss = None, verbose=False):
    """
    Calculate the daily percentage increase of the portfolio, including trading costs.

    Parameters:
        portfolio_weights (dict): Dictionary of {ticker: weight}. Includes "cash" for cash weight.
        start_date (str): Start date for the calculation in 'YYYY-MM-DD' format.
        end_date (str): End date for the calculation in 'YYYY-MM-DD' format.
        trading_cost (float): Cost of trading as a fraction (default: 0.01 for 1%).

    Returns:
        float: Daily percentage increase of the portfolio.
    """

    if stop_loss is None:
        stop_loss = 1000 # Set to a high value to avoid activating the stop loss
    portfolio_weights = state_dict["portfolio"]
    invidual_ticker_pct_change = {}

    for idx, date in enumerate(month_dates):

        date = date.strftime("%Y-%m-%d")
        daily_pct_change = 0
        for ticker, weight in portfolio_weights.items():
            if ticker == "cash":
                # Cash does not generate returns but remains constant
                continue
            
            if ticker not in invidual_ticker_pct_change:
                invidual_ticker_pct_change[ticker] = {"pct_change": 0, "stop_loss_active": False}

            try:
                ticker_percentage_change = pct_change_dict[ticker][date]
            except KeyError:
                # If the date is not in the pct_change_dict, then it is because it was recently added to the Stock market and we do not have the pct. change for that date
                # Although the price data is not present, the funamental data is present, so we can still use the fundamental data to make a decision whether to buy the stock or not
                # I am assuming that the model considers a stock to be a good buy if the fundamental data appears good, and the stock has yet to have it's IPO.
                # Instead, we will set the percentage change to 0
                # if verbose:
                #     print(f"Date {date} not found in pct_change_dict for ticker {ticker}, will use 0% change")
                continue

            if idx == 0:
                # Deduct trading cost on the first price change
                ticker_percentage_change -= trading_cost

            if date == month_dates[-1]:
                # Deduct trading cost on the last price change
                ticker_percentage_change -= trading_cost

            invidual_ticker_pct_change[ticker]["pct_change"] += ticker_percentage_change
        
            # Daily percentage change
            if not invidual_ticker_pct_change[ticker]["stop_loss_active"]:
                daily_pct_change += weight * ticker_percentage_change
            else: 
                pass

            # Check if the stop loss should activate, AFTER the daily percentage change has been calculated
            if invidual_ticker_pct_change[ticker]["pct_change"] < -stop_loss:
                if not invidual_ticker_pct_change[ticker]["stop_loss_active"]:
                    invidual_ticker_pct_change[ticker]["stop_loss_active"] = True
                    daily_pct_change -= weight * trading_cost
                    if verbose:
                        print(f"Stop loss activated for {ticker} at date {date} with percentage change {invidual_ticker_pct_change[ticker]['pct_change']}")

        state_dict["returns"].append(daily_pct_change)  # Store daily returns
        state_dict["dates_of_returns"].append(date)
        state_dict["individual_ticker_pct_change"].append(invidual_ticker_pct_change)

        # Update net value of the portfolio
        state_dict["net_value"] *= (1 + daily_pct_change)
    return state_dict


def organize_lstm_data_as_node_predictions(results_dict, ticker_to_index):
    predictions = results_dict['Prediction']
    ground_truths = results_dict['Ground Truth']
    test_dates = results_dict['Test_Dates']
    tickers = results_dict['Tickers']

    sorted_dates = sorted(list(set(test_dates)))
    reconstructed_predictions = [np.full(len(ticker_to_index), 0.0) for _ in sorted_dates]
    reconstructed_ground_truth = [np.full(len(ticker_to_index), 0.0) for _ in sorted_dates]
    date_to_index = {date: idx for idx, date in enumerate(sorted_dates)}

    # Fill the arrays with predictions
    for pred, target, date, ticker in zip(predictions, ground_truths, test_dates, tickers):
        date_idx = date_to_index[date]  
        ticker_idx = ticker_to_index[ticker]  
        reconstructed_predictions[date_idx][ticker_idx] = pred 
        reconstructed_ground_truth[date_idx][ticker_idx] = target 

    sorted_dates = pd.to_datetime(list(set(test_dates))).sort_values()

    return reconstructed_predictions, reconstructed_ground_truth,  sorted_dates


def select_test_or_val(results_dict, test_val):
    if test_val == "test":
        return results_dict
    elif test_val == "val":
        results_dict["Date"] = results_dict["Val_Dates"]
        results_dict["Prediction"] = results_dict["Val_Prediction"]
        results_dict["Ground Truth"] = results_dict["Val_Ground Truth"]
        return results_dict
    else:
        raise ValueError("Invalid test_val argument. Must be 'test' or 'val'.")



def backtester_monthly(model_name, results_dict, index_to_ticker, ticker_to_index, strategy_function, initial_cash=100, 
                        trading_cost=0.01 , k_stocks=15, test_val="test", verbose=False):
    """
    Run a backtesting process with monthly rebalancing.

    Parameters:
        initial_cash (float): Starting capital in USD.
        results_dict (dict): Dictionary with keys "Date" and "Prediction".
        index_to_ticker (dict): Mapping of indices to stock tickers.
        strategy_function (function): Function that determines stock weights for each month.

    Returns:
        dict: Dictionary containing the history of portfolio changes and remaining cash.
    """
    state = {
        "net_value": initial_cash,
        "portfolio": {},
        "dates_to_rebalance": [],
        "dates_of_returns": [],
        "returns": [],
        "history": []
    }

    results_dict = select_test_or_val(results_dict, test_val)

    if "A3TGCN2" in model_name:
        dates = pd.to_datetime(results_dict["Date"]).sort_values()
        predictions = results_dict["Prediction"]

    elif "LSTM" in model_name:
        predictions, reconstructed_ground_truth, dates = organize_lstm_data_as_node_predictions(results_dict, ticker_to_index)
        results_dict["Prediction"] = predictions
        results_dict["Ground Truth"] = reconstructed_ground_truth
        # print("Dates shape:", dates.shape)
        # print("Dates last value:", dates[-1])
        # print("Predictions length: ", len(predictions))
        # print("Predictions[0].shape: ", predictions[0].shape)
        # print("")

    # mean_pred = np.mean(predictions)
    # std_pred = np.std(predictions)
    weight_scalar = np.percentile(predictions, 80)/2

    monthly_groups = group_dates_by_month(dates)
    # start_date as string, end_date as string for SQL query
    price_start_date = monthly_groups[0][0].strftime("%Y-%m-%d")
    price_end_date = monthly_groups[-1][-1].strftime("%Y-%m-%d")

    pct_change_dict = get_percentage_change_for_backtester(start_date=price_start_date, end_date=price_end_date)
    previous_net_value = initial_cash
    if verbose:
        print("Length of monthly groups: ", len(monthly_groups))
    for month_idx, month_dates in enumerate(monthly_groups):
        if verbose:
            print("Length of month_dates: ", len(month_dates))
        last_date_of_month = month_dates[-1]
        month_start_date = month_dates[0]

        monthly_predictions = get_predictions_for_prior_two_weeks(month_start_date, dates, predictions, index_to_ticker)

        # Updating portfolio composition based on strategy
        new_portfolio = strategy_function(monthly_predictions, state["portfolio"], state["net_value"], min_total_weight_scalar=weight_scalar, k_stocks=k_stocks)
        state["portfolio"] = new_portfolio
        state["individual_ticker_pct_change"] = []

        if verbose:
            try:
                print(f"Portfolio created on {last_date_of_month} for month starting {month_start_date}: ", state["portfolio"])
            except IndexError:
                print(f"Portfolio created on {last_date_of_month}: ", state["portfolio"])

        try:
            next_month_dates = monthly_groups[month_idx + 1]
            calc_returns = True
        except IndexError:
            next_month_dates = [month_dates[-1] + timedelta(days=1)]
            calc_returns = False

        if calc_returns:
            state = calculate_daily_percentage_increase(state, next_month_dates, pct_change_dict, trading_cost=trading_cost, verbose=verbose)
            net_value_inceased = ((state["net_value"] - previous_net_value) / previous_net_value)*100
            if verbose:
                print(f"Net value for {model_name} at {next_month_dates[-1]}:  {state['net_value']}   -   Percentage Increase: {net_value_inceased}% \n")
        else:
            net_value_inceased = 0

        state["history"].append({
            "date_of_prediction": last_date_of_month,
            "start_date": next_month_dates[0],
            "end_date": next_month_dates[-1],
            "amount_of_stocks": len(state["portfolio"]) - 1,
            "net_value": state["net_value"],
            "portfolio": state["portfolio"].copy(),
            "individual_ticker_pct_change": state["individual_ticker_pct_change"].copy()
        })
        previous_net_value = state["net_value"]

    return state



def write_portfolio_csv_history(state, initial_cash, model_name, trading_cost):
    """
    Write the portfolio history to a CSV file.

    Parameters:
        state (dict): Dictionary containing the history of portfolio changes.
        filename (str): Name of the CSV file to write.
    """
    history = state["history"]
    returns = state["returns"]
    dates_of_returns = state["dates_of_returns"]

    if trading_cost == 0:
        trading_cost = "0"
    elif trading_cost == 0.01:
        trading_cost = "1"
    elif trading_cost == 0.0008:
        trading_cost = "8"


    df_items = []
    filename = f"{model_name}_monthly_portfolio_history_cost{trading_cost}.xlsx"
    df = pd.DataFrame(history)
    df["cash_holding"] = df["portfolio"].apply(lambda x: x["cash"])
    df = df[['date_of_prediction','start_date', 'end_date', 'net_value', 'amount_of_stocks', 'cash_holding', 'portfolio', 'individual_ticker_pct_change']]
    # Let the % incrase be from week to week
    df["net_value_%_increase"] = df["net_value"].pct_change() * 100
    df["net_value_%_increase"].iloc[0] = ((df["net_value"].iloc[0] - initial_cash) / initial_cash) * 100
    df.to_excel(f"../Data/Results/{filename}", index=False)


    cur_history_item = history[0]
    history_idx = 0
    df_items = []
    for idx, date in enumerate(dates_of_returns):
        
        prev_initial_cash = initial_cash
        initial_cash += initial_cash * returns[idx]
        df_items.append({
            "Date": date,
            "Returns (%)": returns[idx],
            "Previous Portfolio Value (USD)": prev_initial_cash,
            "Portfolio Value (USD)"         : initial_cash,
            "Portfolio % Increase"          : ((initial_cash - prev_initial_cash) / prev_initial_cash) * 100
            # "Cash Holding"                  : cur_history_item["portfolio"]["cash"],
            # "Portfolio Composition"         : cur_history_item["portfolio"]
        })

        # if date == cur_history_item["end_date"]:
        #     history_idx += 1
        #     if len(history) > history_idx:
        #         cur_history_item = history[history_idx]

    df = pd.DataFrame(df_items)
    filename = f"{model_name}_daily_portfolio_history_cost{trading_cost}.xlsx"
    df.to_excel(f"../Data/Results/{filename}", index=False)




if __name__ == "__main__":

    model_name = "A3TGCN2_5"
    # model_name = "LSTM_0"
    # model_name = "LSTM_4"
    initial_cash = 100
    k_stocks = 25#15

    # for model_name in ["A3TGCN2_7", "A3TGCN2_6", "A3TGCN2_5", "A3TGCN2_4", "A3TGCN2_3", "A3TGCN2_2", "A3TGCN2_1", "LSTM_1", "LSTM_2", "LSTM_3", "LSTM_4"]:
    model_name = "A3TGCN2_5"
    for trading_cost in [0, 0.0008, 0.01]:
        index_to_ticker, ticker_to_index = load_index_to_ticker_dict()
        results = load_results(f"{model_name}_results")
        state = backtester_monthly(
            model_name=model_name,
            results_dict=results,
            index_to_ticker=index_to_ticker,
            ticker_to_index=ticker_to_index,
            initial_cash=initial_cash,
            strategy_function=long_strategy_1,
            trading_cost=trading_cost,
            k_stocks = k_stocks,
            test_val= "test",
            verbose=True
        )

        write_portfolio_csv_history(state, initial_cash, model_name, trading_cost)
        # except Exception as e:
        #     print(f"Error in {model_name} with trading cost {trading_cost}: {e}")
            #     continue


# if __name__ == "__main__":
    # backtesting_df = pd.DataFrame()
    # backtesting_df["Top_K"] = [i for i in range(1, 11,2)]+ [i for i in range(10, 55, 5)]
    # initial_cash = 100
    # trading_cost = 0.0008
    # # trading_cost = 0
    # index_to_ticker, ticker_to_index = load_index_to_ticker_dict()

    # # for model_name in ["A3TGCN2_1", "A3TGCN2_2", "A3TGCN2_3", "A3TGCN2_4", "A3TGCN2_5", "A3TGCN2_6", "A3TGCN2_7", "LSTM_1", "LSTM_2", "LSTM_3", "LSTM_4"]:
    # for model_name in ['A3TGCN2_1','A3TGCN2_2','A3TGCN2_3','A3TGCN2_4','A3TGCN2_5', 'A3TGCN2_6', 'A3TGCN2_7', 'LSTM_1','LSTM_2', 'LSTM_3', 'LSTM_4']:
    #     model_net_values = []

    #     for k_stocks in tqdm(backtesting_df["Top_K"], desc=f"Backtesting {model_name}", total=len(backtesting_df)):
    #         model_results = load_results(f"{model_name}_results")
    #         state = backtester_monthly(
    #                     model_name=model_name,
    #                     results_dict=model_results,
    #                     index_to_ticker=index_to_ticker,
    #                     ticker_to_index=ticker_to_index,
    #                     initial_cash=initial_cash,
    #                     strategy_function=long_strategy_1,
    #                     trading_cost=trading_cost,
    #                     k_stocks = k_stocks,
    #                     test_val= "test",
    #                     verbose=False
    #                 )
    #         model_net_values.append(state["net_value"])
    #     backtesting_df[model_name] = model_net_values
    # backtesting_df.to_excel("../Data/Results/Top_K_Backtesting_Overall_with_cost.xlsx", index=False)

