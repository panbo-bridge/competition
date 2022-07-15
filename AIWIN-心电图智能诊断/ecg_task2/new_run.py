#线下的主程序
import codecs,glob,os
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
from sklearn import preprocessing
import numpy as np
import pandas as pd
import torch
import time
import config
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
from model import TextCNN,ECGNet
# from resnet import resnet34,resnet50
from models2 import myecgnet
from scipy import signal
# from dataset import ECGDataset
from new_dataset import ECGDataset
torch.manual_seed(41)
torch.cuda.manual_seed(41)
BATCH_SIZE = 24
EPOCHS = 80
LEARNING_RATE = 0.0001
show_interval=10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(device)
train_dataset = ECGDataset(train=True)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_dataset = ECGDataset(train=False)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

model = myecgnet().to(device)
model.load_state_dict(torch.load("resnet34_model.mdl",map_location='cpu'))
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)#0.01权重过大
w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
criterion = utils.WeightedMultilabel(w)
def train_epoch(model, optimizer, criterion, train_dataloader, show_interval=10):
    model.train()
    lgb_feature = []
    lgb_label = []
    f1_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(device)
        target = target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(inputs)
        # output_ = output.detach().cpu().numpy()
        # lgb_feature.append(output_)
        # lgb_label.append(target.detach().cpu().numpy())
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        it_count += 1
        f1 = utils.calc_f1(target, torch.sigmoid(output))
        f1_meter += f1
        if it_count != 0 and it_count % show_interval == 0:
            print("%d,loss:%.3e f1:%.3f" % (it_count, loss.item(), f1))
    # np.save("train_feature.npy",lgb_feature)
    # np.save("train_label.npy",lgb_label)
    return loss_meter / it_count, f1_meter / it_count
def val_epoch(model, criterion, val_dataloader, threshold=0.5):
    model.eval()
    lgb_feature = []
    lgb_label = []
    f1_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            print(inputs.shape)
            inputs = inputs.to(device)
            target = target.to(device)
            output = model(inputs)
            output_ = output.detach().cpu().numpy()
            lgb_feature.append(output_)
            lgb_label.append(target.detach().cpu().numpy())

            loss = criterion(output, target)
            loss_meter += loss.item()
            it_count += 1
            output = torch.sigmoid(output)
            f1 = utils.calc_f1(target, output, threshold)
            f1_meter += f1
    lgb_feature = np.array(lgb_feature)
    lgb_label = np.array(lgb_label)
    np.save("test_feature.npy",lgb_feature)
    np.save("test_label.npy",lgb_label)
    return loss_meter / it_count, f1_meter / it_count
#输出预测结果时还需要加1
if __name__ == "__main__":
    best_f1 = -1
    for epoch in range(0, 1):
        since = time.time()
        #train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
        print(val_f1)
        #print('#epoch:%02d  train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
        #      % (epoch, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        #if best_f1 < val_f1:
        #    print(f' f1 imporve from {best_f1} to {val_f1} Save Model...')
        #    torch.save(model.state_dict(), f"task1_model_ECGNet.mdl")
        #    best_f1 = val_f1
###  val = 0.8746

