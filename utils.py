import pandas as pd
from typing import Tuple

def get_dataset(path: str) -> pd.DataFrame:
    """Read the path as csv and returns the data

    Returns:
        pandas.dataFrame
        None on error
    """
    try:
        dataframe = pd.read_csv(path)
    except Exception as e:
        print(f"{get_dataset.__name__}: {e}")
        exit(1)
    return dataframe
