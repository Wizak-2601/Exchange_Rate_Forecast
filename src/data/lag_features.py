import pandas as pd

def add_lags(data, lags):
    df = pd.DataFrame(data)

    for lag in lags:
        df[f"lag_{lag}"] = df.iloc[:, -1].shift(lag)

    df = df.dropna()

    return df.values.astype("float32")
import pandas as pd

def add_lags(data, lags):
    df = pd.DataFrame(data)

    for lag in lags:
        df[f"lag_{lag}"] = df.iloc[:, -1].shift(lag)

    df = df.dropna()

    return df.values.astype("float32")
