from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from pathlib import Path


def fit_scaler(data, save_path=None):
    """
    Fit StandardScaler on training data.
    Accepts numpy arrays or pandas DataFrames.
    """

    if isinstance(data, pd.DataFrame):
        data = data.values

    scaler = StandardScaler()
    scaler.fit(data)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_path)

    return scaler


def transform_data(data, scaler):
    """
    Apply scaler to numpy arrays or DataFrames.
    Returns numpy array.
    """

    if isinstance(data, pd.DataFrame):
        data = data.values

    return scaler.transform(data)