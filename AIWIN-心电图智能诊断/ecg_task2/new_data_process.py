# -*- coding: utf-8 -*-
'''
@time: 2019/9/8 18:44
数据预处理：
    1.构建label2index和index2label
    2.划分数据集
@ author: javis
'''
#线下的迁移学习数据处理
import os, torch
import numpy as np
from config import config

# 保证每次划分数据一致
def valuetrans(value):
    #转换标签
    new_value = []
    for i in value:
        if i == 4:
            new_value.append(0)
        elif i == 2:
            new_value.append(1)
        elif i == 7:
            new_value.append(2)
        elif i == 17:
            new_value.append(3)
        elif i == 13:
            new_value.append(4)
        elif i == 15:
            new_value.append(5)
        elif i == 10:
            new_value.append(6)
        elif i == 20:
            new_value.append(7)
        elif i == 16:
            new_value.append(8)
        elif i == 1:
            new_value.append(9)
        elif i == 3:
            new_value.append(10)
        elif i in [5,6,8,9,11,12,14,18,19,21,22,23,24,25]:
            new_value.append(11)
    new_value = list(set(new_value))
    return new_value
def name2index(path):
    '''
    把类别名称转换为index索引
    :param path: 文件路径
    :return: 字典
    '''
    list_name = []
    for line in open(path):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx
def split_data(file2idx, val_ratio=0.1):
    '''
    划分数据集,val需保证每类至少有1个样本
    :param file2idx:
    :param val_ratio:验证集占总数据的比例
    :return:训练集，验证集路径
    '''

    data = set(os.listdir(config.train_dir))
    val = set()
    idx2file = [[] for _ in range(55)]
    for file, list_idx in file2idx.items():
        for idx in list_idx:
            idx2file[idx].append(file)
    new_idx2file = []
    new_idx2file.append(idx2file[4])#正常
    new_idx2file.append(idx2file[2][0:200])#过缓
    new_idx2file.append(idx2file[7][0:200])#过速
    new_idx2file.append(idx2file[17][0:200])#不齐
    new_idx2file.append(idx2file[13][0:200])#心房颤动
    new_idx2file.append(idx2file[15][0:200])#室性早搏
    new_idx2file.append(idx2file[10][0:200])#房性早搏
    new_idx2file.append(idx2file[20][0:200])#正常一度房室传导阻滞
    new_idx2file.append(idx2file[16][0:200])#完全性右束传导阻滞
    new_idx2file.append(idx2file[1][0:200])#T波改变
    new_idx2file.append(idx2file[3][0:200])#sT波改变   5，6，8，9,11,12
    other = []
    for i in [5,6,8,9,11,12,14,18,19,21,22,23,24,25]:
        for j in idx2file[i]:
            other.append(j)
    new_idx2file.append(other)
    idx2file = new_idx2file
    for item in idx2file:
        # print(len(item), item)
        num = int(len(item) * val_ratio)
        val = val.union(item[:num])
    train = data.difference(val)
    return list(train), list(val)
def file2index(path, name2idx):
    '''
    获取文件id对应的标签类别
    :param path:文件路径
    :return:文件id对应label列表的字段
    '''
    file2index = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        labels = [name2idx[name] for name in arr[3:]]
        # print(id, labels)
        file2index[id] = labels
    return file2index
def count_labels(data, file2idx):
    '''
    统计每个类别的样本数
    :param data:
    :param file2idx:
    :return:
    '''
    cc = [0]*12
    for fp in data:
        for i in file2idx[fp]:
            cc[i] += 1
    return np.array(cc)
def train(name2idx, idx2name):
    file2idx = file2index(config.train_label, name2idx)
    new_file2idx = {}
    for key,value in file2idx.items():
        new_file2idx[key] = valuetrans(value)
    train, val= split_data(file2idx)
    print(new_file2idx)
    wc=count_labels(train,new_file2idx)
    print(wc)
    dd = {'train': train, 'val': val, 'file2idx': new_file2idx,'wc':wc}
    torch.save(dd, config.train_data)
if __name__ == '__main__':
    pass
    name2idx = name2index(config.arrythmia)
    idx2name = {idx: name for name, idx in name2idx.items()}
    train(name2idx, idx2name)