#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import utils


def main():
    parser = argparse.ArgumentParser(description='Make histogram about houses marks distribution per discipline')
    args = parser.parse_args()

    df = utils.get_dataset('./datasets/train.csv')
    houses = df['Hogwarts House'].unique()
    features = utils.get_numeric_features(df, drop_index=True).columns

    size = int(len(features) ** .5) + 1
    fig, axes = plt.subplots(nrows=size, ncols=size, gridspec_kw={'hspace': .5, 'wspace': .2})
    for i, feature in enumerate(features):
        plot = axes[i // size][i % size]
        plot.set_title(feature)
        plot.set_xlabel('Marks')
        plot.set_ylabel('Nº of students')
        plot.set_xticks([])
        plot.set_yticks([])
        for house in houses:
            value = df[df['Hogwarts House'] == house][feature].dropna()
            plot.hist(value, alpha=0.6)
        plot.grid()
    i += 1
    while i % size != 0:
        fig.delaxes(axes[i // size][i % size])
        i += 1
    fig.legend(houses, loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
