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

from SQLite_tools import query_stock_data, query_and_combine_TA_data, create_and_upload_table 
from SQLite_tools import check_if_close_price_exists, fetch_all_trading_days, get_all_quarters, fetch_all_tickers



def construct_graph(timeseries_data, edge_data_dict, ticker_to_index, date, graph_dir, target_data):
    """
    Constructs a PyTorch Geometric graph for a single day, ensuring alignment using ticker_to_index.

    Args:
        timeseries_data (dict): Dictionary with node features for each date.
        edge_data_dict (dict): Dictionary with edge index and attributes for each date.
        ticker_to_index (dict): Mapping from ticker to node index.
        date (str): The date for which the graph is constructed.
        graph_dir (str): Directory to save/load the graph data.
        target_data (pd.DataFrame): DataFrame containing target labels with 'Date' and 'Ticker' as keys.

    Returns:
        Data: A PyTorch Geometric Data object representing the graph.
    """
    day_graph_path = f"{graph_dir}/{date}.pkl"
    if os.path.exists(day_graph_path):
        with open(day_graph_path, 'rb') as f:
            return pickle.load(f)

    # Get node features for the date
    day_data = timeseries_data[date]
    day_data = day_data.set_index('Ticker')

    target_day_data = target_data[target_data['Date'] == date]
    target_day_data = target_day_data.set_index('Ticker')['target']

    # Initialize node features and targets arrays
    num_nodes = len(ticker_to_index)
    node_features = np.zeros((num_nodes, day_data.shape[1]), dtype=np.float32)
    node_targets = np.zeros(num_nodes, dtype=np.float32)

    for ticker, idx in ticker_to_index.items():
        node_features[idx] = day_data.loc[ticker].values
        try:
            node_targets[idx] = target_day_data.loc[ticker]
        except KeyError:
            # This shouldnt happen, but in the off chance it does, we replace with 0
            print(f"Ticker {ticker} not found in target data for {date}, replacing with 0")
            node_targets[idx] = 0
            

    node_features = torch.tensor(node_features, dtype=torch.float)
    node_targets = torch.tensor(node_targets,   dtype=torch.float)

    edge_index = torch.tensor(edge_data_dict[date]["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(edge_data_dict[date]["edge_attributes"], dtype=torch.float)

    graph = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=node_targets)

    with open(day_graph_path, 'wb') as f:
        pickle.dump(graph, f)

    return graph


def create_and_store_chunks(timeseries_data, edge_data_dict, ticker_to_index, window_size, GRAPH_DATA_DIR, CHUNK_DATA_DIR):
    """
    Constructs and stores overlapping chunks of graphs for time-series data.

    Args:
        timeseries_data (dict): Dictionary where keys are dates and values are daily DataFrames.
        edge_data_dict (dict): Dictionary containing edge indices and attributes for each date.
        ticker_to_index (dict): Mapping of tickers to node indices.
        window_size (int): Number of days per chunk.
        GRAPH_DATA_DIR (str): Directory to save/load individual graphs.
        CHUNK_DATA_DIR (str): Directory to save graph chunks.
    """
    sorted_days = fetch_all_trading_days(remove_first_3_months=True)
    num_days = len(sorted_days)
    target_data = query_stock_data(table="Target_Data", ticker="all_stocks")

    for i in tqdm(range(window_size - 1, num_days), desc="Creating and storing chunks. Graphs are loaded from disk if they exist."):
        chunk_days = sorted_days[i - window_size + 1:i + 1]
        chunk_graphs = []
        chunk_filename = f"{CHUNK_DATA_DIR}/chunk_{chunk_days[-1]}__{window_size}.pkl"

        if os.path.exists(chunk_filename):
            continue

        for day in chunk_days:
            graph = construct_graph(
                timeseries_data=timeseries_data,
                edge_data_dict=edge_data_dict,
                ticker_to_index=ticker_to_index,
                target_data=target_data,
                date=day,
                graph_dir=GRAPH_DATA_DIR
            )
            chunk_graphs.append(graph)

        with open(chunk_filename, 'wb') as f:
            pickle.dump(chunk_graphs, f)


def fetch_timeseries_node_data():
    """
    Create a dictionary containing daily node data for all trading days.
    """
    timeseries_node_data = {}
    timeseries_node_data_path = f"../Data/Nodes/timeseries_node_data.pkl"
    if os.path.exists(timeseries_node_data_path):
        with open(timeseries_node_data_path, "rb") as f:
            print(f"Loading precomputed node data from {timeseries_node_data_path}")
            return pickle.load(f)

    all_trading_days = fetch_all_trading_days(remove_first_3_months=True)

    for date in tqdm(all_trading_days, desc="Creating timeseries data"):
        if date not in timeseries_node_data:
            ml_feature_day_data = query_stock_data(table="ML_Feature_Table", ticker= "all_stocks", start_date = date, end_date = date)
            ml_feature_day_data.drop(columns=["Date"], inplace=True)
            timeseries_node_data[date] = ml_feature_day_data.copy(deep=True)

    with open(timeseries_node_data_path, "wb") as f:
        pickle.dump(timeseries_node_data, f)
        print(f"Timeseries node data saved to {timeseries_node_data_path}")

    return timeseries_node_data


def fetch_edge_weights(correlations_long_term, correlations_short_term, ticker1, ticker2):
    try:
        weight1 = correlations_long_term[(ticker1,ticker2)]['correlation']
    except:
        weight1 = 0
    try:
        weight2 = correlations_short_term[(ticker1,ticker2)]['correlation']
    except:
        weight2 = 0

    return weight1, weight2

def correlation_edge_index(ticker_to_index: dict):
    """
    Computes edge indices and single-dimensional edge weights based on correlation data 
    from a precomputed dictionary for an undirected graph.

    Args:
        ticker_to_index (dict): Mapping of tickers to node indices.

    Returns:
        dict: A dictionary containing edge indices and single-dimensional edge weights for each date.
    """
    new_edge_dict_path = f"../Data/Edges/sparse_edges_with_single_weight.pkl"
    if os.path.exists(new_edge_dict_path):
        with open(new_edge_dict_path, "rb") as f:
            print(f"Loading precomputed edge data from {new_edge_dict_path}")
            return pickle.load(f)

    print("Computing edge data...")

    # Load the short-term correlation data
    with open(f"../Data/Edges/correlation_short_term.pkl", "rb") as f:
        edge_dict_short_term = pickle.load(f)

    all_trading_days = fetch_all_trading_days(remove_first_3_months=True)
    all_edge_days = sorted(edge_dict_short_term.keys())
    all_tickers_in_use = fetch_all_tickers()  # Guarantees all tickers with data are included
    edge_data_dict = {}

    for date in tqdm(all_trading_days, desc="Creating sparse edge data"):
        if date not in edge_dict_short_term:
            # Find the closest date in the past with available data
            closest_date = None
            for edge_date in all_edge_days:
                if edge_date < date:
                    closest_date = edge_date
                else:
                    break
            if closest_date:
                correlations_short_term = edge_dict_short_term[closest_date]
            else:
                continue
        else:
            correlations_short_term = edge_dict_short_term[date]

        edge_set = set()   
        edge_weights = []  

        for ticker1, ticker2 in correlations_short_term:
            weight = correlations_short_term[(ticker1, ticker2)]['correlation']
            if weight == 0:
                continue

            if ticker1 in ticker_to_index and ticker2 in ticker_to_index:
                idx1, idx2 = ticker_to_index[ticker1], ticker_to_index[ticker2]
                edge = tuple(sorted((idx1, idx2)))  # Ensure (u, v) == (v, u) by sorting indices
                
                if edge not in edge_set:  # super ensuring to avoid duplicate edges
                    edge_set.add(edge)
                    edge_weights.append(weight)  # Add weight for unique edge


        # Convert edge_set and edge_weights to arrays
        edge_index = np.array(list(edge_set)).T  
        edge_weights = np.array(edge_weights, dtype=np.float32)  

        # Verify alignment
        assert edge_index.shape[1] == edge_weights.shape[0], \
            f"Mismatch: edge_index has {edge_index.shape[1]} edges, but edge_weights has {edge_weights.shape[0]} weights"


        edge_data_dict[date] = {
            "edge_index": edge_index,
            "edge_attributes": edge_weights
        }

    # Save the edge data dictionary for later use
    with open(f"{new_edge_dict_path}", "wb") as f:
        pickle.dump(edge_data_dict, f)
        print(f"Edge data dictionary with single-dimensional weights saved to {new_edge_dict_path}")

    return edge_data_dict



def save_ticker_mappings(ticker_to_index):
    index_to_ticker = {idx: ticker for ticker, idx in ticker_to_index.items()}
    with open("../Data/Nodes/index_to_ticker.pkl", "wb") as f:
        pickle.dump(index_to_ticker, f)
        print("Index to ticker mapping saved to ../Data/Nodes/ticker_to_index.pkl")
    
    with open("../Data/Nodes/ticker_to_index.pkl", "wb") as f:
        pickle.dump(ticker_to_index, f)
        print("Ticker to index mapping saved to ../Data/Nodes/index_to_ticker.pkl")



if __name__ == "__main__":

    WINDOW_SIZE = 5
    CHUNK_DATA_DIR = f"../Data/Networks_chunks/window_size_{WINDOW_SIZE}"
    GRAPH_DATA_DIR = "../Data/Networks"
    os.makedirs(CHUNK_DATA_DIR, exist_ok=True)
    os.makedirs(GRAPH_DATA_DIR, exist_ok=True)
    os.makedirs("../Data/Nodes", exist_ok=True)

    all_tickers_in_use = fetch_all_tickers()  
    ticker_to_index = {ticker: idx for idx, ticker in enumerate(all_tickers_in_use)}

    timeseries_node_data = fetch_timeseries_node_data()
    edge_data_dict = correlation_edge_index(ticker_to_index)

    create_and_store_chunks(timeseries_node_data, edge_data_dict, ticker_to_index, WINDOW_SIZE, GRAPH_DATA_DIR, CHUNK_DATA_DIR)

    save_ticker_mappings(ticker_to_index)



