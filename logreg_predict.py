#!/usr/bin/env python3

import csv
import numpy as np
import utils
import argparse
from typing import Tuple, List, Dict


def save_prediction(y_hat: List[str]) -> None:
    """ Save prediction into file

    Args:
        y_hat (List[str]): List of predicted labels
    """
    keys = ['Index', utils.LABEL_FEATURE]
    rows = [{keys[0]: i, keys[1]: pred} for i, pred in enumerate(y_hat)]
    try:
        with open('houses.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as e:
        print(f"{save_prediction.__name__} failed: {e}")


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
    x, norm_x, thetas = get_data()
    y_hat = []

    # Predict for all labels and concatenate into one matrix
    one_vs_all = []
    for house, theta in thetas.items():
        theta = np.array(theta).reshape(-1, 1)
        one_vs_all.append(utils.predict(norm_x, theta))
    one_vs_all = np.concatenate(one_vs_all, axis=1)

    # Find max-value for each row to determine the predicted label
    labels = list(thetas.keys())
    for pred in one_vs_all:
        index = np.argmax(pred)
        pred_label = labels[index]
        y_hat.append(pred_label)

    save_prediction(y_hat)


def main():
    parser = argparse.ArgumentParser(description='Predict with logistic regression')
    args = parser.parse_args()

    predict()


if __name__ == "__main__":
    main()
