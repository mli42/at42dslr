#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import utils

ROW_INDEX = {
    "Count": 'count',
    "Mean": 'mean',
    "Std": 'std',
    "Min": 'min',
    "25%": 'first_quartile',
    "50%": 'second_quartile',
    "75%": 'third_quartile',
    "Max": 'max',
}


class Describe:
    def __init__(self, df: pd.DataFrame) -> None:
        self.count = 0
        self.mean = 0
        self.std = 0
        self.min = 0
        self.first_quartile = 0
        self.second_quartile = 0
        self.third_quartile = 0
        self.max = 0


def describe(df: pd.DataFrame) -> None:
    numerical_features = df.select_dtypes(include=[np.number])
    if 'Index' in numerical_features.columns:
        numerical_features.drop('Index', axis=1, inplace=True)

    describe_data = {}
    for index, data in numerical_features.iteritems():
        feature_description = Describe(df)
        describe_data[index] = [getattr(feature_description, key) for key in ROW_INDEX.values()]
        break

    description = pd.DataFrame(describe_data, index=ROW_INDEX.keys())
    print(description)


def main():
    parser = argparse.ArgumentParser(description='Train model with linear regression')
    parser.add_argument('dataset', type=str, help='dataset on which "describe" is performed')
    args = parser.parse_args()
    df = utils.get_dataset(args.dataset)
    describe(df)


if __name__ == "__main__":
    main()
