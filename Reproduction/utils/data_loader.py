import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np
import os
from PIL import Image


# Data reading function
def read_data(path, img_folder_path):
    df = pd.read_csv(path)
    df['img_path'] = df['ImageFile'].apply(lambda f: os.path.join(img_folder_path, f))
    return df


# Dataset class for image aesthetics
class AestheticsDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df
        self.is_train = is_train
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = mpimg.imread(row['img_path'])

        if image.ndim == 2:  # 如果图像是单通道（灰度图像）
            image = np.stack((image,) * 3, axis=-1)  # 复制通道，使其变为三通道
        elif image.shape[-1] == 4:  # 如果图像是RGBA格式，转换为RGB格式
            image = image[:, :, :3]

        # 将numpy数组转换为PIL图像对象
        image = Image.fromarray((image * 255).astype(np.uint8))

        image = self.transform(image)  # 应用transforms到PIL图像

        # 只选择数值列作为标签，跳过 'ImageFile' 和 'img_path'
        attributes = ['BalancingElements', 'ColorHarmony', 'Content', 'DoF', 'Light',
                      'MotionBlur', 'Object', 'Repetition', 'RuleOfThirds', 'Symmetry',
                      'VividColor', 'score']
        labels = torch.tensor([float(row[attr]) for attr in attributes], dtype=torch.float32)

        return image, labels


# DataLoader creation function
def create_dataloader(csv_file, img_folder, batch_size=32, shuffle=True, is_train=True):
    df = read_data(csv_file, img_folder)
    dataset = AestheticsDataset(df, is_train=is_train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
