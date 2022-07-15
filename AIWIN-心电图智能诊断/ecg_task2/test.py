import pandas as pd
import glob
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
import sys
import os
import scipy.io as sio
import numpy as np
import torch
from model import TextCNN,ECGNet
from models2 import myecgnet
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
from simple_ecgnet import ECGNet_3
from torch.utils.data import DataLoader,Dataset
import csv
from scipy import signal
def resample(sig, target_point_num=None):
    '''
    对原始信号进行重采样
    :param sig: 原始信号
    :param target_point_num:目标型号点数
    :return: 重采样的信号
    '''
    sig = signal.resample(sig, target_point_num) if target_point_num else sig
    return sig
class TestDataset(Dataset):
    def __init__(self, mat,mat_dim=4000):
        super(TestDataset, self).__init__()
        self.mat = mat
        self.mat_dim = mat_dim
    def __len__(self):
        return len(self.mat)
    def __getitem__(self, index):
        # idx = np.random.randint(0, 5000-self.mat_dim)
        idx = 500
        # idy = np.random.choice(range(12), 9)
        data =self.mat[index][:, idx:idx+self.mat_dim]
        # data = data.transpose()
        # data = resample(data,2000)
        # data = data.transpose()
        # data = min_max_scaler.fit_transform(data)
        # data = resample(data,3840)
        for i in range(data.shape[0]):
            data[i] = preprocessing.scale(data[i])
        data = torch.tensor(data,dtype=torch.float32)
        return data,-1
#加载数据
path = sys.argv[1]
test_path = glob.glob(os.path.join(path,'*.mat'))
#test_path = glob.glob('/datasets/heart/task2/Train/*.mat')
test_path.sort()
test_path = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in test_path]
Test_Loader = DataLoader(TestDataset(np.array(test_path)), batch_size=1, shuffle=False)
#加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
model = myecgnet().to(device)
model.load_state_dict(torch.load(f"model_3fold_best.mdl",map_location='cpu'))
model.eval()
test_pred = []
with torch.no_grad():
    for i, (x, y) in enumerate(Test_Loader):
        x = x.to(device)
        output = torch.sigmoid(model(x)).squeeze().cpu().numpy()
        ixs = [i+1 for i, out in enumerate(output) if out > 0.3]
        test_pred.append(ixs)
    test_path = glob.glob(os.path.join(path,'*.mat'))
    test_path = [os.path.basename(x)[:-4] for x in test_path]
    test_path.sort()
    files = open('answer.csv','w')
    writer = csv.writer(files,quoting=csv.QUOTE_NONE,quotechar=None,escapechar=' ')
    for i in range(len(test_path)):
        label_value = test_path[i]
        for j in test_pred[i]:
            label_value = label_value + ',' + str(j)
        writer.writerow([label_value])
    files.close()


# with open('answer.csv', 'w', encoding='utf-8') as f:
#     for i in range(len(test)):
#         s = ','.join(test[['name'] + [f'ill_{j}' for j in range(1, 13)]].values[i])
#         f.write(s.replace('mask,', '').strip('mask,') + '\n')
#
# for i in range(1, 13):
#     test[f'ill_{i}'] = test[f'ill_{i}'].apply(lambda x: str(i) if x == 1 else 'mask')
#
# with open('answer.csv', 'w', encoding='utf-8') as f:
#     for i in range(len(test)):
#         s = ','.join(test[['name'] + [f'ill_{j}' for j in range(1, 13)]].values[i])
#         f.write(s.replace('mask,', '').strip('mask,') + '\n')