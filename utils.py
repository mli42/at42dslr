import numpy as np
import pandas as pd
from typing import Tuple

def get_numeric_features(df: pd.DataFrame, drop_index: bool = False) -> pd.DataFrame:
    df = df.select_dtypes(include=[np.number])

    if drop_index and 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)
    return df

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
