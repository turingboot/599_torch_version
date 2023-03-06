# -*- coding: utf-8 -*-
from collections import defaultdict

import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.model.s2s.seq2seq import Seq2Seq
from src.utils.data_loader import create_data_loader
from src.model.lstm.train_eval import train, evaluate
from src.model.lstm import lstm
import warnings

warnings.filterwarnings('ignore')

# Set random seeds
BATCH_SIZE = 64
RANDOM_SEED = 42
seq_len = 48*3
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# Defining the equipment for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------Start of dataset processing section-----------------------------------------#
# Importing data sets
df = pd.read_csv("../dataset/train/FLX_AT-Neu_FLUXNET2015_FULLSET_HH_2002-2012_1-4.csv")
# Dividing the data set
df_train, df_test = train_test_split(
    df,
    test_size=0.2,
    shuffle=False
)

df_val, df_test = train_test_split(
    df_test,
    test_size=0.5,
    random_state=0,
    shuffle=False
)
mm = MinMaxScaler()
ss = StandardScaler()
normalize_func = [mm, ss]

train_data_loader = create_data_loader(df_train, seq_len=seq_len, batch_size=BATCH_SIZE,
                                       device=device, normalize_func=normalize_func)
val_data_loader = create_data_loader(df_val, seq_len=seq_len, batch_size=BATCH_SIZE,
                                     device=device, normalize_func=normalize_func)
test_data_loader = create_data_loader(df_test, seq_len=seq_len, batch_size=BATCH_SIZE,
                                      device=device, normalize_func=normalize_func)
# --------------------------------------End of dataset processing section-----------------------------------------#


# --------------------------网络初始化以及网络的训练和验证部分开始------------------------------------#
input_size = 14  # number of features
hidden_size = 256  # number of features in hidden state
num_layers = 1  # number of stacked lstm layers
num_classes = 48*3  # number of output classes
weight_decay = 0.0001  # 权重衰减系数
# 导入网络模型
# sm_model = lstm.LSTM(num_classes, input_size, hidden_size, num_layers, device=device)

sm_model = Seq2Seq(channel_num=input_size, hidden_size=hidden_size, num_layers=num_layers,
                   output_size=num_classes, batch_size=BATCH_SIZE, device=device)

# 将模型转移到训练设备
sm_model = sm_model.to(device)
# 定义损失函数
criterion = nn.MSELoss(reduction='mean')
criterion = criterion.to(device)
# 优化器
lr = 1e-3
optimizer = torch.optim.Adam(sm_model.parameters(), lr=lr,weight_decay=weight_decay)
# 训练的轮数
epochs = 1
best_loss = 99999.0
# 开始训练循环
history = defaultdict(list)
for step in range(epochs):
    train_loss = train(
        model=sm_model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        data_loader=train_data_loader,
        epoch=step,
        epochs=epochs
    )

    val_loss = evaluate(
        model=sm_model,
        data_loader=val_data_loader,
        loss_fn=criterion,
        device=device,
        epoch=step,
        epochs=epochs,
        status='eval'
    )

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

    if val_loss < best_loss:
        torch.save(sm_model.state_dict(), '../saved_models/best_model_state.pth')
        best_loss = val_loss

# loss值可视化
plt.plot(history['train_loss'], label='train loss')
plt.plot(history['val_loss'], label='validation loss')
plt.title('Training history')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.ylim([0, 0.1])
plt.show()

# 测试模型
# test_loss= eval_model(
#   model=sm_lstm,
#   data_loader=test_data_loader,
#   loss_fn=criterion,
#   device=device,
# )
# print(test_loss)

# y_prd, y = get_predictions(model=sm_lstm, data_loader=test_data_loader, device=device)
# plt.plot(y, label="Actual Data")
# plt.plot(y_prd, label="LSTM Predictions")
# plt.savefig("small_plot.png", dpi=300)
# plt.show()
# --------------------------网络初始化以及网络的训练和验证部分结束------------------------------------#
