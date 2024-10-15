# models/cnn_feature_extractor.py
import torch
import torch.nn as nn
import torchvision.models as models


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # Using a pre-trained ResNet model
        resnet = models.resnet50(pretrained=True)
        # Remove the final fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, 2048, H, W]
        return features
