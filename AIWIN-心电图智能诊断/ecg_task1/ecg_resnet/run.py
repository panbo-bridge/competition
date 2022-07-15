import codecs,glob,os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from transformer import Transformer
skf = StratifiedKFold(n_splits=10)
from torchsummary import summary
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from model import TextCNN,ECGNet
from resnet import resnet34,resnet50
train_mat = glob.glob('./train/*.mat')
train_mat.sort()
train_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in train_mat]
train_mat = np.array(train_mat)

# train_mat_1 = train_mat[:,0:2,:]
# print(train_mat_1.shape)
# train_mat_2 = train_mat[:,6:,:]
# print(train_mat_2.shape)
# new_mat = np.concatenate((train_mat_1,train_mat_2),axis=1)
# print(new_mat.shape)
test_mat = glob.glob('./val/*.mat')
test_mat.sort()
test_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in test_mat]
train_df = pd.read_csv('trainreference.csv')
train_df['tag'] = train_df['tag'].astype(np.float32)
train_mat = torch.Tensor(train_mat)
print(train_mat.shape)
# plt.plot(range(5000), train_mat[0][0][0])
# plt.plot(range(5000), train_mat[0][0][1])
# plt.plot(range(5000), train_mat[0][0][3])
# plt.show()
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
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = verflip(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    # 后置不可或缺的步骤
    # sig = sig.transpose()
    sig = torch.tensor(sig.copy(), dtype=torch.float)
    return sig
class MyDataset(Dataset):
    def __init__(self, mat, label, train = False ,mat_dim=5000):
        super(MyDataset, self).__init__()
        self.mat = mat
        self.label = torch.LongTensor(label)
        self.mat_dim = mat_dim
        self.train = train
    def __len__(self):
        return len(self.mat)
    def __getitem__(self, index):
        # idx = np.random.randint(0, 5000-self.mat_dim)
        idx = 0
        # idy = np.random.choice(range(12), 9)
        data =self.mat[index][:, idx:idx+self.mat_dim]
        # data = data.transpose()
        # data = transform(data, self.train)
        # data = data.transpose(1,0)
        label = self.label[index]
        return data,label
# model = TextCNN(2)
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0001
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(device)
fold_idx = 0
for tr_idx, val_idx in skf.split(train_mat, train_df['tag'].values):
    Train_Loader = DataLoader(MyDataset(np.array(train_mat)[tr_idx], np.array(train_df['tag'].values[tr_idx]),train = True), batch_size=BATCH_SIZE, shuffle=True)
    Val_Loader = DataLoader(MyDataset(np.array(train_mat)[val_idx], np.array(train_df['tag'].values[val_idx])), batch_size=BATCH_SIZE, shuffle=True)
    Test_Loader = DataLoader(MyDataset(np.array(test_mat), np.array(train_df['tag'].values[0:400])), batch_size=1, shuffle=False)
    model = resnet34().to(device)
    # model.load_state_dict(torch.load("model_1.mdl"))
    # model = Transformer(10000,12,2,device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    Test_best_Acc = 0
    for epoch in range(0, EPOCHS):
        correct = 0
        total = 0
        Train_Loss, Test_Loss = [], []
        Train_Acc, Test_Acc = [], []
        model.train()
        for i, (x, y) in enumerate(Train_Loader):
            # x = x.squeeze(1)#transformer使用的
            x = x.to(device)
            y = y.to(device)
            pred = model(x.to(torch.float32))
            loss = criterion(pred, y)
            Train_Loss.append(loss.item())
            #
            # pred = (torch.nn.functional.sigmoid(pred)>0.5).astype(int)
            # Train_Acc.append((pred.numpy() == y.numpy()).mean())
            _, label_index = torch.max(pred.data, dim=-1)
            # print(label_index)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
            # correct += (label_index == y.long()).sum().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        Train_Acc = correct/total
        model.eval()
        correct = 0
        total = 0
        for i, (x, y) in enumerate(Val_Loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x.to(torch.float32))
            Test_Loss.append(criterion(pred, y).item())
            # pred = (torch.nn.functional.sigmoid(pred)>0.5).astype(int)
            # Test_Acc.append((pred.numpy() == y.numpy()).mean())
            _, label_index = torch.max(pred.data, dim=-1)
            total += label_index.shape[0]
            correct += (label_index == y.long()).sum().item()
        Test_Acc = correct/total
        if epoch % 1 == 0:
            print(
                "Epoch: [{}/{}] TrainLoss/TestLoss: {:.4f}/{:.4f} TrainAcc/TestAcc: {:.4f}/{:.4f}".format( \
                epoch + 1, EPOCHS, \
                np.mean(Train_Loss), np.mean(Test_Loss), \
                np.mean(Train_Acc), np.mean(Test_Acc) \
                )
            )
        if Test_best_Acc < np.mean(Test_Acc):
            print(f'Fold {fold_idx} Acc imporve from {Test_best_Acc} to {np.mean(Test_Acc)} Save Model...')
            torch.save(model.state_dict(), f"model_{fold_idx}.mdl")
            Test_best_Acc = np.mean(Test_Acc)
    model = resnet34().to(device)
    model.load_state_dict(torch.load(f"model_{fold_idx}.mdl"))
    model.eval()
    test_pred = []
    for i, (x, y) in enumerate(Test_Loader):
        x = x.to(device)
        pred = model(x.to(torch.float32))
        _, label_index = torch.max(pred.data, dim=-1)
        test_pred.append(label_index.item())

    test_path = glob.glob('./val/*.mat')
    test_path = [os.path.basename(x)[:-4] for x in test_path]
    test_path.sort()
    test_answer = pd.DataFrame({
        'name': test_path,
        'tag': test_pred
    }).to_csv('answer_'+ str(fold_idx) + ".csv", index=None)
            # total += label_index.shape[0]
            # correct += (label_index == y.long()).sum().item()
    print("ok")
    fold_idx += 1


