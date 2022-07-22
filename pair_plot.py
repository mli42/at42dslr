#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import utils


def main():
    DEFAULT_HUE = 'Hogwarts House'
    parser = argparse.ArgumentParser(description='Make pair plot between features')
    parser.add_argument('--hue', type=str, default=DEFAULT_HUE, help=f'Feature to map different colors, default is "{DEFAULT_HUE}"')
    args = parser.parse_args()

    df = utils.get_dataset('./datasets/train.csv', drop_index=True)

    if not args.hue in df.columns:
        raise ValueError(f'"{args.hue}" is not a feature of given dataset')
    elif not pd.api.types.is_string_dtype(df[args.hue]):
        raise ValueError(f'"{args.hue}" feature is numerical')

    hue_values = df[args.hue].unique()
    features = utils.get_numeric_features(df).columns
    size = len(features)

    fig, axes = plt.subplots(nrows=size, ncols=size)
    for j, feature_y in enumerate(features):
        for i, feature_x in enumerate(features):
            ax = axes[j][i]

            for single_hue in hue_values:
                df_hue = df[df[args.hue] == single_hue]
                if i == j:
                    ax.hist(df_hue[feature_x].dropna(), alpha=0.6)
                else:
                    ax.scatter(df_hue[feature_x], df_hue[feature_y], edgecolor='white', linewidth=.5)

            if i == 0:
                ax.set_ylabel(feature_y, labelpad=(4. * (3 * (j % 2))))
            else:
                ax.set_yticks([])

            if j == size -1:
                ax.set_xlabel(feature_x, labelpad=(4. * (3 * (i % 2))))
            else:
                ax.set_xticks([])

    fig.suptitle("Pair plot of Hogwarts Houses' marks")
    fig.legend(hue_values, title=args.hue, loc='center right')
    plt.show()


if __name__ == "__main__":
    main()
