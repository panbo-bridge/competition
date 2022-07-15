#resnetresnet进行训练集和测试集的划分
import random
random.seed(41)
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
    cc = [0] * label_num
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)
def split_data(file2idx, data,val_ratio=0.1):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''
    val = set()
    idx2file = [[] for _ in range(label_num)]
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[int(idx)].append(file)
    for item in idx2file:
        #num = int(len(item) * val_ratio)
        num = 10000
        #random.shuffle(item)
        val = val.union(item[:num])
    train = data.difference(val)
    return list(train), list(val)
if __name__ == "__main__":
    label_num = 2 #该属性下标签的种数
    train_df = open('attr_label_2.csv')
    csv_reader_lines = csv.reader(train_df)
    csv_reader_lines2 = csv_reader_lines
    file2idx = {}
    data = set()
    #标签计数
    for item in csv_reader_lines:
        if item[0] == "name":
            continue
        data.add(item[0])
        file2idx[item[0]] = [int(i) for i in item[1:]]
    train,val = split_data(file2idx,data)
    wc=count_labels(train,file2idx)
    dd = {'train': train, 'val': val,'file2idx': file2idx,'wc':wc,"label_num":label_num}
    print(dd["wc"])
    wc=count_labels(val,file2idx)
    print(wc)
    torch.save(dd, "attr_train_2.pth")
