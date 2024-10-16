# utils/metrics.py
import torch
import numpy as np

def mean_squared_error(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)       #评价是不如nn.MSELoss

def pearson_correlation(y_true, y_pred):
    y_true = y_true - torch.mean(y_true)
    y_pred = y_pred - torch.mean(y_pred)
    return torch.sum(y_true * y_pred) / (torch.sqrt(torch.sum(y_true ** 2)) * torch.sqrt(torch.sum(y_pred ** 2)))

# 定义KL散度互信息损失
def mutual_information_loss(p_s_given_a_m, q_s_given_m):
    # 计算 KL 散度，表示两个分布之间的差异
    kl_loss = nn.functional.kl_div(q_s_given_m.log(), p_s_given_a_m, reduction='batchmean')
    return kl_loss