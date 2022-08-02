#!/usr/bin/env python3

import sys
import argparse
import pandas as pd
from typing import List
from sklearn.metrics import accuracy_score


def load_csv(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filename)
        data = df['Hogwarts House']
    except Exception as e:
        sys.exit(f"{e}")
    return data

def main():
    DEFAULT_FILE_A = './datasets/train.csv'
    DEFAULT_FILE_B = './houses.csv'

    parser = argparse.ArgumentParser(description='Calculate the accuracy between two csv')
    parser.add_argument('--file-a', type=str, default=DEFAULT_FILE_A, help=f'CSV used (default: {DEFAULT_FILE_A})')
    parser.add_argument('--file-b', type=str, default=DEFAULT_FILE_B, help=f'CSV used (default: {DEFAULT_FILE_B})')
    args = parser.parse_args()

    file_a = load_csv(args.file_a)
    file_b = load_csv(args.file_b)

    print(f"{file_a.shape = } || {file_b.shape = }")
    if file_a.shape != file_b.shape:
        sys.exit("Different shapes between files")
    score = accuracy_score(file_a, file_b)
    print("Your score: %.3f" % score)

if __name__ == '__main__':
    main()
