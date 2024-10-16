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
        """
        删除最后两层的全局池化和全连接层，让CNNFeatureExtractor的输出是卷积层生成的特征图(feature map)，而不是分类结果
        Layer Name: 8
        Layer Structure: AdaptiveAvgPool2d(output_size=(1, 1))
        
        Layer Name: 9
        Layer Structure: Linear(in_features=2048, out_features=1000, bias=True)
        AdaptiveAvgPool2d(output_size=(1, 1)) 会将H * W * C特征图缩小1×1×C，在输入全连接层时会自动展平为 C 维的向量，因此可以直接传递给全连接层，不需要额外的操作
        """

    def forward(self, x):
        features = self.feature_extractor(x)  # [B, 2048, H, W]
        return features

"""
resnet50 = models.resnet50(pretrained=True)
for name, layer in enumerate(resnet50.children()):
    print(f"Layer Name: {name}\nLayer Structure: {layer}\n")
"""