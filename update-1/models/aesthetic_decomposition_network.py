# models/aesthetic_decomposition_network.py
import torch
import torch.nn as nn


class AestheticDecompositionNetwork(nn.Module):
    def __init__(self, feature_dim=2048, num_attributes=11):
        super(AestheticDecompositionNetwork, self).__init__()
        # 定义一个三层的全连接网络
        self.fc1 = nn.Linear(feature_dim, 102)  # 第一隐藏层，有102个单元
        self.fc2 = nn.Linear(102, 19)           # 第二隐藏层，有19个单元
        self.fc3_weights = nn.Linear(19, num_attributes)  # 输出层，用于预测各个属性的权重
        #self.fc3_biases = nn.Linear(19, num_attributes)   # 输出层，用于预测各个属性的偏置
        self.relu = nn.ReLU()  # ReLU激活函数
        self.softmax = nn.Softmax(dim=1)  # Softmax函数用于归一化权重

    def forward(self, image_features):
        # 通过全局平均池化将特征图展平成一维向量
        x = torch.mean(image_features, dim=[2, 3])

        # 通过全连接层进行特征变换
        x = self.relu(self.fc1(x))  # 第一隐藏层，使用ReLU激活函数
        x = self.relu(self.fc2(x))  # 第二隐藏层，使用ReLU激活函数

        # 基于图像特征生成各个属性的权重（和偏置可选）
        weights = self.softmax(self.fc3_weights(x))  # 生成各属性的权重（w1, w2, ..., wn）
        #biases = self.fc3_biases(x)  # 生成各属性的偏置（b1, b2, ..., bn）

        return weights

