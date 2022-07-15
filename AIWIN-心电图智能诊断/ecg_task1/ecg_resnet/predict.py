import codecs,glob,os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
from torchsummary import summary
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from model import TextCNN,ECGNet
from resnet import resnet34
device = torch.device("cuda")
test_mat = glob.glob('./train/*.mat')
test_mat.sort()
test_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in test_mat]
test_mat = np.array(test_mat)
#
# test_mat_1 = test_mat[:,0:2,:]
# print(test_mat_1.shape)
# test_mat_2 = test_mat[:,6:,:]
# print(test_mat_2.shape)
# test_mat = np.concatenate((test_mat_1,test_mat_2),axis=1)
test_pred = []
test_mat = torch.Tensor(test_mat)

class MyDataset(Dataset):
    def __init__(self, mat, mat_dim=5000):
        super(MyDataset, self).__init__()
        self.mat = torch.LongTensor(mat)
        # self.label = torch.LongTensor(label)
        self.mat_dim = mat_dim
    def __len__(self):
        return len(self.mat)
    def __getitem__(self, index):
        # idx = np.random.randint(0, 5000-self.mat_dim)
        idx = 0
        # idy = np.random.choice(range(12), 9)
        data =self.mat[index][:, idx:idx+self.mat_dim]
        # label = self.label[index]
        return data
Test_Loader = DataLoader(MyDataset(np.array(test_mat)), batch_size=1, shuffle=False)
model = resnet34().to(device)
model.load_state_dict(torch.load("model_0.mdl"))
for i, (x) in enumerate(Test_Loader):
    x = x.to(device)
    pred = model(x.to(torch.float32))
    _, label_index = torch.max(pred.data, dim=-1)
    print(label_index)



for i in range(400):
    # idx = np.random.randint(0, 5000-3000)
    idx = 0
    x = test_mat[i][:, idx:idx+5000]
    x = x.unsqueeze(0)
    print(x.shape)
    result_list = []
    print("当前预测的个数为",i)
    for j in range(1):
        model_name = "model_" + str(j) + ".mdl"
        model.load_state_dict(torch.load(model_name))
        pred = model(x.to(torch.float32))
        # print(pred)
        _, label_index = torch.max(pred.data, dim=-1)
        label = label_index.item()
        result_list.append(label)
        print(label)
    if result_list.count(1) >=3:
        x_label = 1
    else:
        x_label = 0
    test_pred.append(x_label)
test_path = glob.glob('./val/*.mat')
test_path = [os.path.basename(x)[:-4] for x in test_path]
test_path.sort()
test_answer = pd.DataFrame({
    'name': test_path,
    'tag': test_pred
}).to_csv('answer.csv', index=None)



