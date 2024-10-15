# utils/metrics.py
import torch
import numpy as np

def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def pearson_correlation(y_true, y_pred):
    y_true = y_true - torch.mean(y_true)
    y_pred = y_pred - torch.mean(y_pred)
    return torch.sum(y_true * y_pred) / (torch.sqrt(torch.sum(y_true ** 2)) * torch.sqrt(torch.sum(y_pred ** 2)))
