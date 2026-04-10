import pandas as pd
from typing import Protocol

def load_data(path: str) -> pd.DataFrame:
    """
    Loads a dataset from a given CSV file path using pandas.

    Parameters:
    path (str): The absolute or relative path to the CSV file.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the loaded data.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {path} was not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading data: {e}")
