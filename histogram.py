#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


def main():
    df = utils.get_dataset('./datasets/train.csv')
    houses = df['Hogwarts House'].unique()
    features = utils.get_numeric_features(df, drop_index=True).columns

    subplot_len = int(len(features) ** .5) + 1
    fix, axes = plt.subplots(nrows=subplot_len, ncols=subplot_len, gridspec_kw={'hspace': .5, 'wspace': .2})
    for i, feature in enumerate(features):
        plot = axes[i // subplot_len][i % subplot_len]
        plot.set_title(feature)
        plot.set_xlabel('Marks')
        plot.set_ylabel('Nº of students')
        for house in houses:
            value = df[df['Hogwarts House'] == house][feature].dropna()
            plot.hist(value, alpha=0.6)
        plot.legend(houses)
        plot.grid()
    plt.show()


if __name__ == "__main__":
    main()
