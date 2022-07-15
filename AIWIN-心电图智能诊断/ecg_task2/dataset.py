# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 19:47

@ author: javis
'''
#线上的数据处理
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
import pywt, os, copy
import torch
import numpy as np
import pandas as pd
# from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
import scipy.io as sio
torch.manual_seed(41)
torch.cuda.manual_seed(41)
def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
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
    sig = resample(sig, 2048)
    # # 数据增强
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    # 后置不可或缺的步骤
    sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig


class ECGDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """
    def __init__(self,train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load("train.pth")
        self.train = train
        self.data = dd['train'] if train else dd['val']
        # self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])

    def __getitem__(self, index):
        fid = self.data[index]
        file_path = "/datasets/heart/task2/Train/" + fid + ".mat"
        # df = pd.read_csv(file_path, sep=' ').values
        df = sio.loadmat(file_path)['ecgdata'].reshape(12, 5000)
#       x = transform(df, self.train)
#         df = df.transpose()
#         x = resample(df,2000)
#         x = x.transpose()
        id = 1
        if self.train:
            id = np.random.randint(0,3)
        if id == 0:
            idx = 0
        if id == 1:#nohup python -u run.py > nohup.out 2>&1 &
            idx = 500
        if id == 2:
            idx = 1000
        # idy = np.random.choice(range(12), 9)
        x = df[:,idx:idx+4000]
        for i in range(x.shape[0]):
            x[i] = preprocessing.scale(x[i])
        target = np.zeros(12)
        target[self.file2idx[fid]] = 1
        x = torch.tensor(x,dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return x, target
    def __len__(self):
        return len(self.data)
if __name__ == '__main__':
    d = ECGDataset()
    print(d[0])