import os 
import shutil

if __name__ == "__main__":

    # If you read this code, you are probably wondering why i would delete the entire database.
    # Well... it took a few tries to get everything right :)

    # Operating path 
    relative_path = "../Data"

    # All files in the directory that need to be deleted before data processing
    files = ["Edges/correlation_long_term.pkl", "Edges/correlation_short_term.pkl",
            "EOD_financial_data.db", "expected_reports.pkl", "Edges/sparse_edges_with_multi_weights.pkl",
            "Nodes/timeseries_node_data.pkl"]

    for file in files:
        try:
            os.remove(os.path.join(relative_path, file))
            print(f"Deleted {file}")
        except FileNotFoundError:
            print(f"Tried to delete {file}, but File not found")


    # All directories which should be deleted before data processing
    directories = ["Nodes", "Edges", "Networks", "Networks_chunks"]
    for directory in directories:
        try:
            shutil.rmtree(os.path.join(relative_path, directory))
            print(f"Deleted {directory}")
        except FileNotFoundError:
            print(f"Tried to delete {directory}, but Directory not found")