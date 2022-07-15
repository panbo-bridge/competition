import csv
#csv文件，是一种常用的文本格式，用以存储表格数据，很多程序在处理数据时会遇到csv格式文件
import glob
import scipy.io as sio
import numpy as np
train_mat = glob.glob('/datasets/heart/task2/Train/*.mat')
train_mat.sort()
#train_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in train_mat]
files=open('answer.csv','w')
writer=csv.writer(files)
for i in range(1):
    data = sio.loadmat(train_mat[i])['ecgdata'].reshape(12, 5000)
    for j in range(12):
        writer.writerow(data[j])#写入一行
files.close()