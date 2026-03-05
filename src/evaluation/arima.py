import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def smape_numpy(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denominator == 0, 0, diff / denominator))


def arima_multivariate(train_array, val_array, order=(1,1,1)):
    """
    Runs ARIMA independently per feature (column)
    train_array: (T_train, F)
    val_array:   (T_val, F)
    """

    smapes = []

    n_features = train_array.shape[1]

    for i in range(n_features):

        train_series = train_array[:, i]
        val_series   = val_array[:, i]

        model = ARIMA(train_series, order=order)
        fitted = model.fit()

        forecast = fitted.forecast(steps=len(val_series))

        score = smape_numpy(val_series, forecast)
        smapes.append(score)

    return float(np.mean(smapes))
