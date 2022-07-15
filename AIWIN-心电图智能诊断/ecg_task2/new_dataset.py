# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 19:47

@ author: javis
'''
import pywt, os, copy
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import Dataset
from sklearn.preprocessing import scale
from scipy import signal
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()



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
    print(sig.shape)
    sig = resample(sig, config.target_point_num)
    print(sig.shape)
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

    def __init__(self, train=True):
        super(ECGDataset, self).__init__()
        dd = torch.load("data/train_task1.pth")
        self.train = train
        self.data = dd['train'] if train else dd['val']
        # self.idx2name = dd['idx2name']
        self.file2idx = dd['file2idx']
        self.wc = 1. / np.log(dd['wc'])
        self.train_hand_feature = []
        self.count = 0
    def __getitem__(self, index):
        fid = self.data[index]
        file_path = os.path.join(config.train_dir, fid)
        df = pd.read_csv(file_path, sep=' ')  #单个样本（5000,8）通道顺序不一样我擦
        df['III'] = df['II'] - df['I']
        df['aVR'] = -(df['I'] + df['II']) / 2
        df['aVL'] = df['I'] - df['II'] / 2
        df['aVF'] = df['II'] - df['I'] / 2
        # 方法2 ；在原DF上进行修改
        df = df.loc[:,['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']]
        df = df.values
        x = df.transpose()
        # x = transform(df, self.train)
        id = 1
        self.count = self.count + 1
        if self.train:
            id = np.random.randint(0,3)
        if id == 0:
            idx = 0
        if id == 1:#nohup python -u run.py > nohup.out 2>&1 &
            idx = 500
        if id == 2:
            idx = 1000
        # idy = np.random.choice(range(12), 9)
        x = x[:,idx:idx+4000]
        x = x*4.88/1000
        # x = min_max_scaler.fit_transform(x)
        hand_feature = []
        x_min = np.min(x,axis=1)
        x_max = np.max(x,axis=1)
        x_mean = np.mean(x,axis=1)
        x_std = np.std(x,axis=1)
        x_median = np.median(x,axis=1)
        hand_feature.append(x_min)
        hand_feature.append(x_max)
        hand_feature.append(x_mean)
        hand_feature.append(x_std)
        hand_feature.append(x_median)
        self.train_hand_feature.append(hand_feature)
        # train_hand_feature.append()
        self.train_hand_feature_ = np.array(self.train_hand_feature)
        # self.train_hand_feature_ = self.train_hand_feature_.reshape(-1,)
        print(self.train_hand_feature_.shape)
        np.save("test_hand_feature.npy",self.train_hand_feature_)
        # x_skew = np.skew(x,axis=1)
        for i in range(x.shape[0]):
            x[i] = preprocessing.scale(x[i])
        #     print(x[i])
        x = torch.tensor(x, dtype=torch.float32)
        target = np.zeros(config.num_classes)
        target[self.file2idx[fid]] = 1
        target = torch.tensor(target, dtype=torch.float32)
        return x, target

    def __len__(self):
        return len(self.data)
if __name__ == '__main__':
    d = ECGDataset(config.train_data)
    for i in range(2):
        d[i][0]
    # print(d[0][1])