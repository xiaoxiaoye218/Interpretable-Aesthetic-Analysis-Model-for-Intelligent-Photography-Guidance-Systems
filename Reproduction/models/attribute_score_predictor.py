import torch
import torch.nn as nn
from models.attention_module import SpatialAttention

class AttributeScorePredictorWithAttention(nn.Module):
    def __init__(self, num_attributes):
        super(AttributeScorePredictorWithAttention, self).__init__()
        self.num_attributes = num_attributes
        self.attribute_predictors = nn.ModuleList([nn.Linear(2048, 1) for _ in range(num_attributes)])  # 假设ResNet输出2048维特征
        self.attention_modules = nn.ModuleList([SpatialAttention() for _ in range(num_attributes)])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 添加全局平均池化

    def forward(self, features):
        attribute_scores = []
        attention_maps = []

        for i in range(self.num_attributes):
            attention_map = self.attention_modules[i](features)
            attended_features = features * attention_map
            pooled_features = self.global_avg_pool(attended_features)  # 使用全局平均池化
            pooled_features = pooled_features.view(pooled_features.size(0), -1)  # 展平
            score = self.attribute_predictors[i](pooled_features)  # 使用全连接层进行预测
            attribute_scores.append(score)
            attention_maps.append(attention_map)

        # 将所有属性的分数拼接起来
        attribute_scores = torch.cat(attribute_scores, dim=1)  # Shape: [batch_size, num_attributes]
        return attribute_scores, attention_maps
