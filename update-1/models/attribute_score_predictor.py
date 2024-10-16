import torch
import torch.nn as nn
from models.attention_module import SpatialAttention

class AttributeScorePredictorWithAttention(nn.Module):
    def __init__(self, num_attributes):
        super(AttributeScorePredictorWithAttention, self).__init__()
        self.num_attributes = num_attributes
        #ModuleList：并行地处理
        self.attribute_predictors = nn.ModuleList([nn.Linear(2048, 1) for _ in range(num_attributes)])  # 假设ResNet输出2048维特征
        self.attention_modules = nn.ModuleList([SpatialAttention() for _ in range(num_attributes)])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 添加全局平均池化，将每个通道的特征图压缩成一个值(B,C,H,W)-->(B,C)

    def forward(self, features):        #feature: (B,2048,H,W)
        attribute_scores = []
        attention_maps = []

        #同一份输入特征，对每个属性挨个预测 score、attention_map
        for i in range(self.num_attributes):
            attention_map = self.attention_modules[i](features)
            attended_features = features * attention_map
            pooled_features = self.global_avg_pool(attended_features)  # 使用全局平均池化-->(B,2048)
            pooled_features = pooled_features.view(pooled_features.size(0), -1)  # 展平，将除了第0维(Batch)以外的
            score = self.attribute_predictors[i](pooled_features)  # 使用全连接层进行预测
            attribute_scores.append(score)
            attention_maps.append(attention_map)

        # 将所有属性的分数拼接起来
        # 假设我们不考虑batch_size维度，就是把一个tensor的list-->tensor
        attribute_scores = torch.cat(attribute_scores, dim=1)  # Shape: [batch_size, num_attributes]
        return attribute_scores, attention_maps
