#线上的数据处理

import numpy as np
np.random.seed(41)
import scipy.io as sio
import glob
import numpy as np
import pandas as pd
import csv
import torch
def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0] * 12
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)
def split_data(file2idx, data,val_ratio=0.2):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    val = set()
    idx2file = [[] for _ in range(12)]
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[int(idx)-1].append(file)
    for item in idx2file:
        # print(len(item), item)
        num = int(len(item) * val_ratio)
        val = val.union(item[:num])
    train = data.difference(val)
    return list(train), list(val)
train_mat = glob.glob('/datasets/heart/task2/Train/*.mat')
train_mat.sort()
train_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in train_mat]
train_mat = np.array(train_mat)
train_df = open('/datasets/heart/task2/trainreference.csv')
csv_reader_lines = csv.reader(train_df)
csv_reader_lines2 = csv_reader_lines
label_count = [0]*12
file2idx = {}
data = set()
#标签计数
for item in csv_reader_lines:
    label = [0]*12
    file2idx[item[0]] = [int(i)-1 for i in item[1:]]
    data.add(item[0])
    for i in range(1,len(item)):
        label[int(item[i])-1] = 1
    for i in range(12):
        label_count[i] = label_count[i] + label[i]
train,val = split_data(file2idx,data)
wc=count_labels(train,file2idx)
dd = {'train': train, 'val': val,'file2idx': file2idx,'wc':wc}
print(dd["wc"])
torch.save(dd, "train.pth")