import codecs,glob,os
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
from sklearn import preprocessing
import numpy as np
import pandas as pd
import torch
import csv
import random
from sklearn.metrics import f1_score
min_max_scaler = preprocessing.MinMaxScaler()
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
#from torchsummary import summary
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
# from preprocessing import *
#from model import TextCNN,ECGNet
#from resnet import resnet34,resnet50
# train_mat = glob.glob('./train/*.mat')
# train_mat.sort()
from models2 import myecgnet
from scipy import signal
def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num,axis=1) if target_point_num else sig
    return sig

def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise

def verflip(sig):
    '''
    信号竖直翻转
    :param sig:
    :return:
    '''
    return sig[::-1, :]

def shift(sig, interval=20):
    '''
    上下平移
    :param sig:
    :return:
    '''
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset
    return sig


def transform(sig, train=False):
    # 前置不可或缺的步骤
    # sig = resample(sig, 2048)
    # # 数据增强
    # if train:
    #     if np.random.randn() > 0.2: sig = scaling(sig)
    #     # if np.random.randn() > 0.5: sig = verflip(sig)
    #     if np.random.randn() > 0.2: sig = shift(sig)
    # 后置不可或缺的步骤
    # sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig
class MyDataset(Dataset):
    def __init__(self, mat, label, train = False ,mat_dim=1000):
        super(MyDataset, self).__init__()
        self.mat = mat
        self.label = torch.LongTensor(label)
        self.mat_dim = mat_dim
        self.train = train
    def __len__(self):
        return len(self.mat)
    def __getitem__(self, index):
        idx =self.mat[index]
        data = np.load("../ecg_16000_filter_processed/" + idx + ".npy")
        data = data.transpose()
        start = random.randint(0,data.shape[1]-16000)
        data = data[:,start:start + 16000]
        label = self.label[index]
        return data,label,idx
BATCH_SIZE = 48
EPOCHS = 30
LEARNING_RATE = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(device)
fold_idx = 0
pred_result = {}
def label_process(label,index):
    label = label.replace(",","")
    label = label.replace("\"","")
    return int(label[index])
def val():
    dd = torch.load("train_2class.pth")
    train_label = open('train_label_1217.csv')
    csv_reader_lines = csv.reader(train_label)
    train_mat = []
    test_mat = []
    train_df = []
    test_df = []
    pred_list = []
    real_list = []
    #
    cls = 17
    for item in csv_reader_lines:
        if item[0] in dd["val"]:
            test_mat.append(item[0])
            test_df.append(label_process(item[1],cls))
    Val_Loader = DataLoader(MyDataset(np.array(test_mat), np.array(test_df),train = False), batch_size=48, shuffle=True)
    for i, (x, y,idx) in enumerate(Val_Loader):
        x = x.to(device)
        y = y.to(device)
        model = myecgnet().to(device)
        model.load_state_dict(torch.load("model_" + str(cls) + ".mdl"))
        model.eval()
        pred = model(x.to(torch.float32))
        # Test_Loss.append(criterion(pred, y).item())
        # pred = torch.nn.functional.sigmoid(pred)
        # pred = pred[:,1] > 0.45
        # print(pred)
        # Test_Acc.append((pred.numpy() == y.numpy()).mean())
        _, label_index = torch.max(pred.data, dim=-1)
        # total += label_index.shape[0]
        # correct += (label_index == y.long()).sum().item()
        label_index = label_index.cpu().detach().numpy()
        # pred_result[idx] = label_index
    # np.save(cls + ".npy",pred_result)
    #pred = (torch.nn.functional.sigmoid(pred)>0.5).astype(int)
        y = y.cpu().detach().numpy()
        for item in label_index:
            pred_list.append(item)
        for item in y:
            real_list.append(item)
        for i in range(label_index.shape[0]):
            pred_result[idx[i]] = label_index[i]
    np.save(str(cls) + ".npy",pred_result)
    score = f1_score(real_list,pred_list)
    print(score)
if __name__ == "__main__":
    val()
