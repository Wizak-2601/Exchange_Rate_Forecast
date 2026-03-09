import pandas as pd

def add_lags(data, lags):

    df = pd.DataFrame(data)

    lagged_frames = [df]

    for lag in lags:
        lagged = df.shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in df.columns]
        lagged_frames.append(lagged)

    df = pd.concat(lagged_frames, axis=1)

    df = df.dropna()

    return df.values.astype("float32")