import json
import numpy as np
import pandas as pd
from typing import Tuple

THETA_FILE = "theta.json"
FEATURES = ["Astronomy", "Ancient Runes"]
THETA_SIZE = len(FEATURES) + 1
LABEL_FEATURE = "Hogwarts House"


def add_intercept(x: np.ndarray) -> np.ndarray:
    """ Adds a column of 1's to the non-empty numpy.ndarray x.
    Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
    Returns:
        x': a matrix of dimension m * (n + 1).
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    ones = np.ones((x.shape[0], 1))
    res = np.concatenate((ones, x), axis=1)
    return res


def sigmoid(x: np.ndarray) -> np.ndarray:
    """ Compute the sigmoid of a vector.
    Args:
        x: an numpy.ndarray, a vector
    Returns:
        The sigmoid value as a numpy.ndarray.
        None if x is an empty numpy.ndarray.
    """
    if x.size == 0:
        return None
    return 1 / (1 + np.exp(-x))

def predict(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """ Computes the vector of prediction y_hat.
    Args:
        x: an numpy.ndarray, a vector of dimension m * n.
        theta: an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
    """
    if (x.shape[1] + 1) != theta.shape[0]:
        return None
    x = add_intercept(x)
    y_hat = sigmoid(x.dot(theta))
    return y_hat


def get_numeric_features(df: pd.DataFrame, drop_index: bool = False) -> pd.DataFrame:
    df = df.select_dtypes(include=[np.number])
    df.dropna(how='all', axis=1, inplace=True)

    if drop_index and 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)
    return df


def get_dataset(path: str, drop_index: bool = False) -> pd.DataFrame:
    """Read the path as csv and returns the data

    Returns:
        pandas.dataFrame
        None on error
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"{get_dataset.__name__}: {e}")
        exit(1)
    if drop_index and 'Index' in df.columns:
        df.drop('Index', axis=1, inplace=True)
    return df


def str_array(array: np.ndarray) -> str:
    return str(array).replace('\n', '')


def save_theta(theta: np.ndarray) -> None:
    string = json.dumps(theta, sort_keys=True, indent=4)
    try:
        with open(THETA_FILE, "w") as file:
            file.write(string + '\n')
    except Exception as e:
        print(f"{save_theta.__name__} failed: {e}")


def get_default_theta() -> np.ndarray:
    return np.asarray([[.0]] * THETA_SIZE)

def get_theta() -> np.ndarray:
    theta = None

    try:
        with open(THETA_FILE, "r") as file:
            theta = json.load(file)
    except Exception as e:
        print(f"File `{THETA_FILE}` corrupted: {e}")
    return theta


def minmax(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
        x: has to be an numpy.ndarray, m * 1.
    Returns:
        x' as a numpy.ndarray, m * 1.
    """
    span = x_max - x_min
    res = (x - x_min) / span
    return res


def get_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read the data and returns the data X and the labels Y
    Returns:
        Tuple[np.ndarray, np.ndarray]: x , y
        None on error
    """
    print(f"Getting data from file '{path}'")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"{get_data.__name__}: {e}")
        exit(1)
    df = df[[*FEATURES, LABEL_FEATURE]].dropna()
    x = df[FEATURES].to_numpy()
    y = df[LABEL_FEATURE].to_numpy().reshape(-1, 1)

    return (x, y)
