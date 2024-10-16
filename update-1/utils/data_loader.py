import pandas as pd
import matplotlib.image as mpimg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import numpy as np
import os
from PIL import Image

# 数据读取函数，根据csv表的内容——把图像的完整路径以及csv表加载到 dataframe里； 其中一行就对应一个图像
# path:csv文件路径， img_folder_path:图片路径
def read_data(path, img_folder_path):
    # 读取CSV文件中的数据
    df = pd.read_csv(path)
    # 生成图像文件的完整路径，并添加到DataFrame的'img_path'列中
    df['img_path'] = df['ImageFile'].apply(lambda f: os.path.join(img_folder_path, f))
    return df

# 自定义的用于图像美学评分的Dataset类
class AestheticsDataset(Dataset):
    # 初始化函数，接收DataFrame数据，是否用于训练
    def __init__(self, df, is_train=True):
        self.df = df  # 保存DataFrame数据
        self.is_train = is_train  # 标记是否为训练数据
        # 定义图像的预处理方法，包括调整尺寸、转换为Tensor以及标准化
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，增加数据多样性。
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 图像标准化
                                     std=[0.229, 0.224, 0.225])  # 使用ImageNet标准的均值和标准差
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 图像标准化
                                     std=[0.229, 0.224, 0.225])  # 使用ImageNet标准的均值和标准差
            ])

    # 返回数据集的长度
    def __len__(self):
        return len(self.df)  # 数据集的样本数量

    # 根据索引获取数据，返回图像和对应的标签
    def __getitem__(self, idx):
        # 获取DataFrame中指定索引的数据
        row = self.df.iloc[idx]
        # 使用matplotlib读取图像，读取出来的是一个numpy数组
        image = mpimg.imread(row['img_path'])

        # 判断图像像素值是否在 [0, 1] 范围内，如果是，将其转换为 [0, 255]
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)

        # 如果图像是单通道（灰度图像），则将其转换为三通道（RGB）
        if image.ndim == 2:  # 如果图像是灰度图
            image = np.stack((image,) * 3, axis=-1)  # 将灰度图扩展为3通道
        # 如果图像是RGBA格式（带透明通道），则去掉透明通道，只保留RGB三通道
        elif image.shape[-1] == 4:  # 如果图像是RGBA
            image = image[:, :, :3]  # 只保留前三个通道（RGB）

        # 将numpy数组转换为PIL图像对象，以便后续应用transforms
        image = Image.fromarray(image)  # 这里不再需要乘以255，因为已经处理过了

        # 对图像应用预处理操作（调整尺寸、转换为Tensor、标准化）
        image = self.transform(image)

        # 选择标签列，跳过 'ImageFile' 和 'img_path' 列，获取与图像对应的美学属性标签
        attributes = ['BalancingElements', 'ColorHarmony', 'Content', 'DoF', 'Light',
                      'MotionBlur', 'Object', 'Repetition', 'RuleOfThirds', 'Symmetry',
                      'VividColor', 'score']
        # 将标签数据转换为PyTorch的FloatTensor格式
        labels = torch.tensor([float(row[attr]) for attr in attributes], dtype=torch.float32)

        # 返回图像Tensor和标签Tensor
        return image, labels


# DataLoader创建函数，用于批量加载数据
def create_dataloader(csv_file, img_folder, batch_size=32, shuffle=True, is_train=True):
    # 读取CSV文件并生成图像路径
    df = read_data(csv_file, img_folder)  # 这一步会根据csv里的文件名去img_folder下找到对应的图片集合，加载到DataFrame里来
    # 创建自定义的Dataset对象
    dataset = AestheticsDataset(df, is_train=is_train)
    # 创建DataLoader，用于批量加载数据
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)      #此shuffle是代表对整个数据集打乱，而不是一个batch内
