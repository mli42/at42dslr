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

    describe_data = {}
    for index, data in numerical_features.iteritems():
        describe_data[index] = [getattr(Math, key)(data) for key in ROW_INDEX.values()]

    description = pd.DataFrame(describe_data, index=ROW_INDEX.keys())
    return description


def main():
    parser = argparse.ArgumentParser(description='Describe given dataset')
    parser.add_argument('dataset', type=str, help='dataset on which "describe" is performed')
    parser.add_argument('--show-real', action='store_true', help='print true output of pandas.describe()')
    args = parser.parse_args()

    df = utils.get_dataset(args.dataset)
    description = describe(df)

    # pd.set_option('display.max_columns', None)
    print(description)
    if args.show_real:
        print(df.select_dtypes(include=[np.number]).describe())


if __name__ == "__main__":
    main()
