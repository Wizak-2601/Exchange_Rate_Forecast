import numpy as np

def create_windows(data, seq_len, pred_len):

    X, Y = [], []

    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])

    return np.array(X), np.array(Y)
