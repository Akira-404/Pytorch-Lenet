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
# from tensorboardX import SummaryWriter

'''
训练lenet
'''

# 设置随机种子


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed()
rmb_label = {"1": 0, "100": 1}

# 设置参数
max_epoch = 10
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

train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

# 构建dataloder
train_loader = DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size)


net = LeNetSequetial(classes=2)
net.initialize_weights()

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
# 设置学习率下降策略
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

train_curve = list()
valid_curve = list()

iter_count = 0

# writer = SummaryWriter(comment='test_your_comment',
#                        filename_suffix="_test_your_filename_suffix")

for epoch in range(max_epoch):
    loss_mean = 0.
    correct = 0.
    total = 0.

    net.train()

    for i, data in enumerate(train_loader):
        iter_count += 1

        # 前向传播
        inputs, labels = data
        outputs = net(inputs)

        # 反向传播
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().numpy()

        loss_mean += loss.item()
        train_curve.append(loss.item())

        if (i + 1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print(
                "Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch,
                    max_epoch,
                    i +
                    1,
                    len(train_loader),
                    loss_mean,
                    correct /
                    total))
            loss_mean = 0.

    scheduler.step()  # 每个 epoch 更新学习率
    # 每个 epoch 计算验证集得准确率和loss
    # validate the model
    if (epoch + 1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                inputs, labels = data
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).squeeze().sum().numpy()

                loss_val += loss.item()

            valid_curve.append(loss_val / valid_loader.__len__())
            print(
                "Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch,
                    max_epoch,
                    j +
                    1,
                    len(valid_loader),
                    loss_val,
                    correct_val /
                    total_val))

#打印图像信息
train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
# 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
valid_x = np.arange(1, len(valid_curve) + 1) * train_iters * val_interval
valid_y = valid_curve

plt.plot(train_x, train_y, label='Train')
plt.plot(valid_x, valid_y, label='Valid')

plt.legend(loc='upper right')
plt.ylabel('loss value')
plt.xlabel('Iteration')
plt.show()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
test_dir = os.path.join(BASE_DIR, "rmb_test_data")

test_data = RMBDataset(data_dir=test_dir, transform=valid_transform)
valid_loader = DataLoader(dataset=test_data, batch_size=1)

for i, data in enumerate(valid_loader):
    # 前向传播
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)

    rmb = 1 if predicted.numpy()[0] == 0 else 100
    print("模型获得{}元".format(rmb))
