import torch
from .metrics import smape

def compute_naive(X, Y, device):
    X = torch.tensor(X, dtype=torch.float32).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).to(device)

    last_value = X[:, -1:, :]
    naive_pred = last_value.repeat(1, Y.shape[1], 1)

    return smape(Y, naive_pred).item()
