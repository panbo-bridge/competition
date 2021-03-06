# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:25:58 2020
文本分类 双向LSTM + Attention 算法
@author: 
"""
import pandas as pd
import numpy as np
import scipy.io as sio
import glob
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_processor import DataProcessor
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
torch.manual_seed(123) #保证每次运行初始化的随机数相同

embedding_size = 4500   #词向量维度
num_classes = 2    #二分类
sentence_max_len = 12  #单个句子的长度
hidden_size = 256

num_layers = 8  #一层lstm
num_directions = 2  #双向lstm
lr = 0.0005
batch_size = 64
epochs = 1000
train_mat = glob.glob('./train/*.mat')
train_mat.sort()
train_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in train_mat]
test_mat = glob.glob('./val/*.mat')
test_mat.sort()
test_mat = [sio.loadmat(x)['ecgdata'].reshape(12, 5000) for x in test_mat]
train_df = pd.read_csv('trainreference.csv')
train_df['tag'] = train_df['tag'].astype(np.float32)
train_mat = torch.Tensor(train_mat)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyDataset(Dataset):
    def __init__(self, mat, label, mat_dim=4500):
        super(MyDataset, self).__init__()
        self.mat = torch.LongTensor(mat)
        self.label = torch.LongTensor(label)
        self.mat_dim = mat_dim
    def __len__(self):
        return len(self.mat)
    def __getitem__(self, index):
        idx = np.random.randint(0, 5000-self.mat_dim)
        # idx = 0
        # idy = np.random.choice(range(12), 9)
        data =self.mat[index][:, idx:idx+self.mat_dim]
        label = self.label[index]
        return data,label
#Bi-LSTM模型
class BiLSTMModel(nn.Module):
    def __init__(self, embedding_size,hidden_size, num_layers, num_directions, num_classes):
        super(BiLSTMModel, self).__init__()
        
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = num_directions
        
        
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers = num_layers, bidirectional = (num_directions == 2))
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.liner = nn.Linear(hidden_size, num_classes)
        # self.act_func = nn.Softmax(dim=1)
    
    def forward(self, x):
        #lstm的输入维度为 [seq_len, batch, input_size]
        #x [batch_size, sentence_length, embedding_size]
        x = x.permute(1, 0, 2)         #[sentence_length, batch_size, embedding_size]
        
        #由于数据集不一定是预先设置的batch_size的整数倍，所以用size(1)获取当前数据实际的batch
        batch_size = x.size(1)
        
        #设置lstm最初的前项输出
        h_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(device)
        
        #out[seq_len, batch, num_directions * hidden_size]。多层lstm，out只保存最后一层每个时间步t的输出h_t
        #h_n, c_n [num_layers * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        
        #将双向lstm的输出拆分为前向输出和后向输出
        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out  #[seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  #[batch, seq_len, hidden_size]
        
        #为了使用到lstm最后一个时间步时，每层lstm的表达，用h_n生成attention的权重
        h_n = h_n.permute(1, 0, 2)  #[batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1) #[batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  #[batch, hidden_size]
        
        attention_w = self.attention_weights_layer(h_n)  #[batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch, 1, hidden_size]
        
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  #[batch, 1, seq_len]
        softmax_w = F.softmax(attention_context, dim=-1)  #[batch, 1, seq_len],权重归一化
        
        x = torch.bmm(softmax_w, out)  #[batch, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch, hidden_size]
        x = self.liner(x)
        # x = self.act_func(x)
        return x
        
def test(model, test_loader, loss_func):
    model.eval()
    loss_val = 0.0
    corrects = 0.0
    for datas, labels in test_loader:
        datas = datas.to(device)
        labels = labels.to(device)
        
        preds = model(datas.to(torch.float32))
        loss = loss_func(preds, labels)
        
        loss_val += loss.item() * datas.size(0)
        
        #获取预测的最大概率出现的位置
        preds = torch.argmax(preds, dim=1)
        corrects += torch.sum(preds == labels.long()).item()
    test_loss = loss_val / len(test_loader.dataset)
    test_acc = corrects / len(test_loader.dataset)
    print("Test Loss: {}, Test Acc: {}".format(test_loss, test_acc))
    return test_acc

def train(model, train_loader,test_loader, optimizer, loss_func, epochs):
    best_val_acc = 0.0
    best_model_params = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        model.train()
        loss_val = 0.0
        corrects = 0.0
        for datas, labels in train_loader:
            datas = datas.to(device)
            labels = labels.to(device)
            
            preds = model(datas.to(torch.float32))
            loss = loss_func(preds, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_val += loss.item() * datas.size(0)
            
            #获取预测的最大概率出现的位置
            preds = torch.argmax(preds, dim=1)
            corrects += torch.sum(preds == labels.long()).item()
        train_loss = loss_val / len(train_loader.dataset)
        train_acc = corrects / len(train_loader.dataset)
        if(epoch % 2 == 0):
            print("Train Loss: {}, Train Acc: {}".format(train_loss, train_acc))
            test_acc = test(model, test_loader, loss_func)
            if(best_val_acc < test_acc):
                best_val_acc = test_acc
                best_model_params = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_params)
    return model

# processor = DataProcessor()
# train_datasets, test_datasets = processor.get_datasets(vocab_size=vocab_size, embedding_size=embedding_size, max_len=sentence_max_len)

for tr_idx, val_idx in skf.split(train_mat, train_df['tag'].values):
    train_loader = DataLoader(MyDataset(np.array(train_mat)[tr_idx], np.array(train_df['tag'].values[tr_idx])), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(MyDataset(np.array(train_mat)[val_idx], np.array(train_df['tag'].values[val_idx])), batch_size=batch_size, shuffle=True)
    model = BiLSTMModel(embedding_size, hidden_size, num_layers, num_directions, num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()
    model = train(model, train_loader, test_loader, optimizer, loss_func, epochs)
    print("---------------------")


