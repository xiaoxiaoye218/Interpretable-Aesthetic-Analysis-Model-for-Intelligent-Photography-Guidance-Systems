# models/aesthetic_decomposition_network.py
import torch
import torch.nn as nn


class AestheticDecompositionNetwork(nn.Module):
    def __init__(self, feature_dim=2048, num_attributes=11):
        super(AestheticDecompositionNetwork, self).__init__()
        # Decomposition Network (Hyper-network) to learn attribute weights
        self.fc1 = nn.Linear(feature_dim, 128)
        self.fc2 = nn.Linear(128, num_attributes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, image_features):
        x = torch.mean(image_features, dim=[2, 3])  # Global Average Pooling
        x = torch.relu(self.fc1(x))
        weights = self.softmax(self.fc2(x))  # Attribute weights (w1, w2, ..., wn)
        return weights
