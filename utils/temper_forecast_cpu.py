import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time

# 自定义数据集类
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(MyDataset, self).__init__()
        self.X = X  # 特征数据
        self.y = y  # 标签数据

    def __getitem__(self, index):
        X, y = torch.Tensor(self.X[index]), torch.Tensor([self.y[index]])
        return X, y

    def __len__(self):
        return len(self.X)

# 神经网络模型类
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(3, 128)  # 第一个全连接层
        self.bn1 = nn.BatchNorm1d(128)   # 第一层的Batch Normalization
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout，防止过拟合

        self.layer2 = nn.Linear(128, 256)  # 第二个全连接层
        self.bn2 = nn.BatchNorm1d(256)   # 第二层的Batch Normalization

        self.layer3 = nn.Linear(256, 128)  # 第三个全连接层
        self.bn3 = nn.BatchNorm1d(128)   # 第三层的Batch Normalization

        self.layer4 = nn.Linear(128, 1)  # 输出层

    def forward(self, x):
        # 定义前向传播过程
        x = F.relu(self.bn1(self.layer1(x)))  # 激活函数ReLU + BatchNorm1
        x = self.dropout1(x)  # Dropout
        x = F.relu(self.bn2(self.layer2(x)))  # 激活函数ReLU + BatchNorm2
        x = F.relu(self.bn3(self.layer3(x)))  # 激活函数ReLU + BatchNorm3
        x = self.layer4(x)  # 输出层
        return x

# 绘制验证损失曲线
def val_plot(total_loss):
    x = range(len(total_loss))
    plt.plot(x, total_loss, label='Val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Val_loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Val_loss.png')

if __name__ == "__main__":
    EPOCH = 100  # 训练轮数
    LR = 0.001  # 学习率
    BATCH_SIZE = 10  # 批处理大小

    # 加载数据
    df = pd.read_csv('tran_data.csv')  # 假设数据文件名为 tran_data.csv
    X = df[[col for col in df.columns if col != 'Pressure' and col != 'Temperture' and col != 'Humidity']].values.astype(float)  # 选取特征列
    y = df['Pressure'].values.astype(float)  # 标签列

    # 数据划分：80%训练数据，10%验证数据，10%测试数据
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=23)

    # 数据标准化：使数据的均值为0，方差为1
    XMean = np.nanmean(X_train_val, axis=0)
    XStd = np.nanstd(X_train_val, axis=0)
    X_train_val = (X_train_val - XMean) / XStd

    # 归一化：特征值缩放到0到1之间
    XMin = np.nanmin(X_train_val, axis=0)
    XMax = np.nanmax(X_train_val, axis=0)
    X_train_val = (X_train_val - XMin) / (XMax - XMin)

    # 划分训练集和验证集：9:1的比例
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, random_state=23)

    # 标准化测试数据：使用训练集的均值和标准差
    X_test = (X_test - XMean) / XStd
    XMin = np.nanmin(X_test, axis=0)
    XMax = np.nanmax(X_test, axis=0)
    X_test = (X_test - XMin) / (XMax - XMin)

    print(f"训练集样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]}")
    print(f"验证集样本数: {X_val.shape[0]}, 特征数: {X_val.shape[1]}")
    print(f"测试集样本数: {X_test.shape[0]}, 特征数: {X_test.shape[1]}")

    # 创建数据集
    train_data = MyDataset(X_train, y_train)
    val_data = MyDataset(X_val, y_val)
    test_data = MyDataset(X_test, y_test)

    # 创建数据加载器
    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = DNN()
    print(model)  # 打印模型结构

    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=LR)  # Adam优化器

    val_MSE = []  # 存储每个epoch的验证集损失

    # 训练循环
    time_start = time.time()
    for epoch in range(EPOCH):
        model.train()  # 设置模型为训练模式
        train_loss = 0.0
        for step, (data, label) in enumerate(train_loader):
            output = model(data)  # 模型预测
            loss = criterion(output, label)  # 计算损失
            optimizer.zero_grad()  # 清除梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_loss += loss.item()

            if step % 10 == 9:  # 每10步打印一次训练损失
                print(f"[{epoch+1}, {step+1}] loss: {train_loss / 10:.3f}")
                train_loss = 0.0

        # 验证
        model.eval()  # 设置模型为评估模式
        val_loss = 0.0
        with torch.no_grad():  # 不计算梯度
            for data, label in val_loader:
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()
            val_MSE.append(val_loss / X_val.shape[0])  # 记录验证集损失

        model.train()  # 训练模式恢复

        # 保存最佳模型
        if len(val_MSE) == 0 or val_MSE[-1] <= min(np.array(val_MSE)):
            print(f"第 {epoch} 轮，验证集MSE最小，保存最佳模型")
            torch.save(model.state_dict(), "Regression-best.th")

    # 绘制验证集损失曲线
    val_plot(val_MSE)

    time_end = time.time()
    print(f'训练时间: {time_end - time_start:.2f} s')

    # 测试阶段
    model.load_state_dict(torch.load('Regression-best.th'))  # 加载最佳模型

    # 预测并保存结果
    DataSet = [['Month', 'Day', 'Hour', 'Pressure', 'forecastPressure']]
    for i in range(X_test.shape[0]):
        for_y = model(torch.tensor(X_test[i]).to(torch.float32))  # 预测
        DataSet.append([X_test[i][0], X_test[i][1], X_test[i][2], y_test[i], for_y.item()])

    with open('forecast_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(DataSet)

    # 计算测试集的损失
    test_loss = 0.0
    with torch.no_grad():
        for data, label in test_loader:
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
    print(f"最佳模型在测试集上的MSE: {test_loss / X_test.shape[0]:.4f}")
    print('测试完成')
