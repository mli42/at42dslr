import numpy as np
import pandas as pd
from typing import Tuple

THETA_FILE = "theta.txt"
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
    print(f"{save_theta.__name__}: saving {str_array(theta)}")
    try:
        with open(THETA_FILE, "w") as file:
            for element in theta:
                file.write(f"{element[0]}\n")
    except Exception as e:
        print(f"{save_theta.__name__} failed: {e}")


def get_theta() -> np.ndarray:
    theta = np.asarray([[.0]] * THETA_SIZE)

    try:
        tmp_theta = theta.copy()
        with open(THETA_FILE, "r") as file:
            for i, line in enumerate(file):
                if i >= THETA_SIZE:
                    raise Exception("File too long")
                tmp_theta[i][0] = float(line.strip())
                if np.isnan(tmp_theta[i][0]):
                    raise Exception("Has NaN")
            if i < THETA_SIZE - 1:
                raise Exception("File too short")
        theta = tmp_theta
    except Exception as e:
        print(f"File `{THETA_FILE}` corrupted: {e}")
        save_theta(theta)
    return theta


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
    x = df[FEATURES].to_numpy()
    y = df[LABEL_FEATURE].to_numpy().reshape(-1, 1)

    return (x, y)
