import torch

def smape(y_true, y_pred, eps=1e-8):
    numerator = torch.abs(y_pred - y_true)
    denominator = torch.abs(y_true) + torch.abs(y_pred) + eps
    return torch.mean(2.0 * numerator / denominator)
