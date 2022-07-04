import numpy as np
import pandas as pd

class Math:

    @staticmethod
    def count(df: pd.DataFrame) -> int:
        return len(df[~df.isna()])

    @staticmethod
    def mean(df: pd.DataFrame) -> float:
        return Math.sum(df) / Math.count(df)

    @staticmethod
    def std(df: pd.DataFrame) -> float:
        res = 0
        m = Math.count(df)
        mean = Math.mean(df)

        for x in df:
            if not np.isnan(x):
                res += (x - mean) ** 2
        res = (res / m) ** 0.5
        return res

    @staticmethod
    def min(df: pd.DataFrame) -> float:
        if Math.count(df) == 0:
            raise ValueError('Empty DataFrame')
        local_min = df[0]
        for x in df:
            if x < local_min:
                local_min = x
        return local_min

    @staticmethod
    def max(df: pd.DataFrame) -> float:
        if Math.count(df) == 0:
            raise ValueError('Empty DataFrame')
        local_max = df[0]
        for x in df:
            if x > local_max:
                local_max = x
        return local_max

    @staticmethod
    def quartile(df: pd.DataFrame, n: float) -> float:
        df = df.sort_values()
        index: float = np.ceil(Math.count(df) * (n / 4)) - 1

        return df[index]

    @staticmethod
    def first_quartile(df: pd.DataFrame) -> float:
        return Math.quartile(df, 1)

    @staticmethod
    def second_quartile(df: pd.DataFrame) -> float:
        return Math.quartile(df, 2)

    @staticmethod
    def third_quartile(df: pd.DataFrame) -> float:
        return Math.quartile(df, 3)

    @staticmethod
    def sum(df: pd.DataFrame) -> float:
        res = 0
        for x in df:
            if not np.isnan(x):
                res += x
        return res
