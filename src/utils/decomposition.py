import pandas as pd
import numpy as np


def moving_average(series, kernel):
    return pd.Series(series).rolling(
        window=kernel,
        center=True
    ).mean()


def seasonal_decompose(series, kernel):
    trend = moving_average(series, kernel)
    seasonal = series - trend
    return trend, seasonal
