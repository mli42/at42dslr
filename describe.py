#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
from Math import Math
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

def describe(df: pd.DataFrame) -> pd.DataFrame:
    numerical_features = df.select_dtypes(include=[np.number])
    if 'Index' in numerical_features.columns:
        numerical_features.drop('Index', axis=1, inplace=True)

    describe_data = {}
    for index, data in numerical_features.iteritems():
        describe_data[index] = [getattr(Math, key)(data) for key in ROW_INDEX.values()]

    description = pd.DataFrame(describe_data, index=ROW_INDEX.keys())
    return description


def main():
    parser = argparse.ArgumentParser(description='Train model with linear regression')
    parser.add_argument('dataset', type=str, help='dataset on which "describe" is performed')
    args = parser.parse_args()

    df = utils.get_dataset(args.dataset)
    description = describe(df)

    pd.set_option('display.max_columns', None)
    print(description)


if __name__ == "__main__":
    main()
