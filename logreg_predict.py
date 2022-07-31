#!/usr/bin/env python3

import numpy as np
import utils
import argparse
from typing import Tuple


def get_data() -> Tuple:
    """ Retrieve data and verify format
    Returns:
        Tuple: (x, norm_x, labels)
        x: np.ndarray of true dataset
        norm_x: np.ndarray of x-normalized dataset
        labels: Dict[str, List[float]] matching label and thetas
    """
    x, _ = utils.get_data("./datasets/test.csv", isTest=True)
    data = utils.get_theta()
    if (
        (data is None) or
        (list(data.keys()) != [utils.LABEL_FEATURE, 'std'])
    ):
        print("File corrupted")
        exit(1)
    labels = data.pop(utils.LABEL_FEATURE)
    std = data.pop('std')
    if (
        not all([isinstance(obj, list) for obj in [*labels.values(), *std.values()]]) or
        len(labels) != 4 or
        len(std) != 2 or
        not all([len(lst) == 3 for lst in labels.values()]) or
        not all([len(lst) == 2 for lst in std.values()]) or
        not all([isinstance(obj, float) for lst in [*labels.values(), *std.values()] for obj in lst])
    ):
        print("File corrupted")
        exit(1)

    # Normalize x-values with minmax from trainset
    norm_x = x.copy()
    for i, norm in enumerate(std.values()):
        x_min, x_max = norm
        norm_x[..., i] = utils.minmax(x[..., i], x_min, x_max)
    return (x, norm_x, labels)

def predict() -> None:
    x, norm_x, labels = get_data()

def main():
    parser = argparse.ArgumentParser(description='Predict with logistic regression')
    args = parser.parse_args()

    predict()


if __name__ == "__main__":
    main()
