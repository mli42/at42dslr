#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import utils
import seaborn as sns


def main():
    df = utils.get_dataset('./datasets/train.csv', drop_index=True)

    g = sns.pairplot(df, hue='Hogwarts House', height=.9, aspect=1.5)
    g.fig.suptitle("Pair plot of Hogwarts Houses' marks", y=1.)
    plt.show()


if __name__ == "__main__":
    main()
