import os
import random
import numpy as np
import torch
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from model.lenet import LeNet, LeNetSequetial
from tools.my_dataset import RMBDataset
# from tensorboardX import  SummaryWriter


# 设置随机种子
def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()
rmb_label = {"1": 0, "100": 1}

# 设置参数
max_epoch = 100
batch_size = 16
lr = 0.01
log_interval = 10
val_interval = 1

# 获取项目根目录
ABSPATH = os.path.abspath(sys.argv[0])
ABSPATH = os.path.dirname(ABSPATH)
print(ABSPATH)
rmb_split_dir = os.path.join(ABSPATH, "rmb_split")
train_dir = os.path.join(rmb_split_dir, "train")
valid_dir = os.path.join(rmb_split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

# 数据增强和转化
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
# 设置验证集的数据增强和转化，不需要 RandomCrop
valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

train_data = RMBDataset(data_dir=train_dir, transforms=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transforms=valid_transform)

train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)
valid_loader=DataLoader(dataset=valid_data,batch_size=batch_size)

net=LeNetSequetial(classes=2)