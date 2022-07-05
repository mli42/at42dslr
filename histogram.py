#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils


def main():
    df = utils.get_dataset('./datasets/train.csv')
    df = utils.get_numeric_features(df, drop_index=True)


if __name__ == "__main__":
    main()
