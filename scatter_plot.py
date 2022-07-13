#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


def main():
    df = utils.get_dataset('./datasets/train.csv')
    features = utils.get_numeric_features(df, drop_index=True).columns
    size = len(features)

    fig = plt.figure()
    for i, feature_x in enumerate(features):
        left_x = size - i
        row = i + 1
        for j, feature_y in enumerate(features[row:], i):
            left_y = size - j

            pos_x = (j + left_x) / (2 * size - 1) + j * 4e-2 - .5
            pos_y = left_x / size - .08

            ax = fig.add_axes([pos_x, pos_y, .05, .05])
            ax.scatter(df[feature_x], df[feature_y], marker=".", alpha=.6)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlabel(feature_x)
            ax.set_ylabel(feature_y)
            # print(i, j, feature_x, "||", feature_y)
    # print(features)
    plt.show()


if __name__ == "__main__":
    main()
