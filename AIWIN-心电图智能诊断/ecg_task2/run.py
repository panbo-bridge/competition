import codecs,glob,os
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
from sklearn import preprocessing
import numpy as np
import pandas as pd
import torch
import time
min_max_scaler = preprocessing.MinMaxScaler()
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import utils
# from transformer import Transformer
skf = StratifiedKFold(n_splits=10)
# from torchsummary import summary
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
# from model import TextCNN,ECGNet
# from resnet import resnet34,resnet50
from models2 import myecgnet
from scipy import signal
from dataset import ECGDataset
# from new_dataset import ECGDataset
from simple_ecgnet import ECGNet_3
torch.manual_seed(41)
torch.cuda.manual_seed(41)
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
show_interval=10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(device)
best_f1 = -1
train_dataset = ECGDataset(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = ECGDataset(train=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = myecgnet().to(device)
model.load_state_dict(torch.load("resnet34_model.mdl",map_location='cpu'))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)#0.01权重过大
w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
criterion = utils.WeightedMultilabel(w)
def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    f1_meter, loss_meter, it_count = 0, 0, 0
    i = 0
    for inputs, target in train_dataloader:
        print("第{}个batch".format(i))
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        i = i + 1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
    return loss_meter / it_count, f1_meter / it_count
def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    f1_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1 = utils.calc_f1(target, output, threshold)
            f1_meter += f1
    return loss_meter / it_count, f1_meter / it_count

for epoch in range(0, EPOCHS):
    since = time.time()
    train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
    val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
    torch.save(model.state_dict(), f"model_3fold_{epoch}.mdl")
    print('#epoch:%02d  train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
          % (epoch, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
    if best_f1 < val_f1:
        print(f' f1 imporve from {best_f1} to {val_f1} Save Model...')
        torch.save(model.state_dict(), f"model_3fold_best.mdl")
        best_f1 = val_f1
#输出预测结果时还需要加1






