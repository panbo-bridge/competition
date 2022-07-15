#线下的主程序
import codecs,glob,os
import warnings
warnings.filterwarnings("ignore", message="Numerical issues were encountered ")
from sklearn import preprocessing
import numpy as np
import pandas as pd
import torch
import time
import config
min_max_scaler = preprocessing.MinMaxScaler()
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
import utils
from sklearn.metrics import f1_score
from model_one import myecgnet_one
# from transformer import Transformer
skf = StratifiedKFold(n_splits=10)
# from torchsummary import summary
import torch.nn as nn
import scipy.io as sio
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset
from model import TextCNN,ECGNet
from resnet import resnet34,resnet50
from models2 import myecgnet
from scipy import signal
# from dataset import ECGDataset
from new_dataset import ECGDataset
from new_dataset import test_Dataset
import random
random.seed(2)
torch.manual_seed(41)
torch.cuda.manual_seed(41)
BATCH_SIZE = 10
EPOCHS = 50
LEARNING_RATE = 0.0001
show_interval=10
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")  # 选择设备 CPU or GPU
print(device)
val_dataset = ECGDataset(train=False)
val_dataloader = DataLoader(val_dataset, batch_size=1)
test_dataset = test_Dataset(train=False)
test_dataloader = DataLoader(test_dataset, batch_size=1)

model = myecgnet().to(device)
model.load_state_dict(torch.load("shandong_myecgnet_best.mdl"))
# model_1 = myecgnet_one().to(device)
# model_1.load_state_dict(torch.load("signal_model/model_1.mdl"))
# model_6 = myecgnet_one().to(device)
# model_6.load_state_dict(torch.load("signal_model/model_6.mdl"))
# model_7 = myecgnet_one().to(device)
# model_7.load_state_dict(torch.load("signal_model/model_7.mdl"))
# model_9 = myecgnet_one().to(device)
# model_9.load_state_dict(torch.load("signal_model/model_9.mdl"))
# model_13 = myecgnet_one().to(device)
# model_13.load_state_dict(torch.load("signal_model/model_13.mdl"))
# model_15 = myecgnet_one().to(device)
# model_15.load_state_dict(torch.load("signal_model/model_15.mdl"))
model_17 = myecgnet_one().to(device)
model_17.load_state_dict(torch.load("signal_model/model_17.mdl"))


# model.load_state_dict(torch.load("resnet34_model.mdl"))
#optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)#0.01权重过大
#w = torch.tensor(train_dataset.wc, dtype=torch.float).to(device)
#criterion = utils.WeightedMultilabel(w)
def test_epoch(model, test_dataloader, threshold=0.5):
    model.eval()
    lgb_feature = []
    lgb_label = []
    predict_result = []
    real_list = []
    pred_list = []
    f1_meter, loss_meter, it_count = 0, 0, 0
    i = 0
    j = 0
    k = 0
    with torch.no_grad():#测试集多了个fid
        for inputs, target in test_dataloader:
            # print(inputs.shape)
            print("第多少次",i)
            i = i + 1
            inputs = inputs.to(device)
            target = target.to(device)
            #fid = fid[0]
            pred_label = []
            # output_1 = model_1(inputs.to(torch.float32))
            # _,label_1 = torch.max(output_1.data,dim=-1)
            # label_1 = label_1.cpu().detach().numpy()[0]


            # output_6 = model_6(inputs)
            # _,label_6 = torch.max(output_6.data,dim=-1)
            # label_6 = label_6.cpu().detach().numpy()[0]
            #
            # output_7 = model_7(inputs)
            # _,label_7 = torch.max(output_7.data,dim=-1)
            # label_7 = label_7.cpu().detach().numpy()[0]
            #
            # output_9 = model_9(inputs)
            # _,label_9 = torch.max(output_9.data,dim=-1)
            # label_9 = label_9.cpu().detach().numpy()[0]
            #
            # output_13 = model_13(inputs)
            # _,label_13 = torch.max(output_13.data,dim=-1)
            # label_13 = label_13.cpu().detach().numpy()[0]
            #
            # output_15 = model_15(inputs)
            # _,label_15 = torch.max(output_15.data,dim=-1)
            # label_15 = label_15.cpu().detach().numpy()[0]
            #
            output_17 = model_17(inputs.to(torch.float32))
            _,label_17 = torch.max(output_17.data,dim=-1)
            label_17 = label_17.cpu().detach().numpy()[0]
            pred_list.append(label_17)

            # output = model(inputs)
            # output = torch.sigmoid(output)
            # threshhold = np.array(threshold)
            # output_ = output.cpu().detach().numpy() > 0.5
            # output_ = output_.astype(int)
            target_ = target.cpu().detach().numpy() > 0.5
            target_ = target_.astype(int)[0]
            real_list.append(target_[17])
            #print(target_)
            # output_ = output_[0]
            # pred_label.append(output_[0])#0
            # pred_label.append(label_1)#1
            # pred_label.append(output_[1])#2
            # pred_label.append(output_[2])#3
            # pred_label.append(output_[3])#4
            # pred_label.append(output_[4])#5

            # pred_label.append(label_6)#6
            # pred_label.append(label_7)#7
            # pred_label.append(output_[5])#8
            # pred_label.append(label_9)#9
            # pred_label.append(output_[6])#10
            # pred_label.append(output_[7])#11
            # pred_label.append(output_[8])#12
            # pred_label.append(label_13)#13
            # pred_label.append(output_[9])#14
            # pred_label.append(label_15)#15
            # pred_label.append(output_[10])#16
            # pred_label.append(label_17)#17)
            # output_ = pred_label
            # real_list.append(target_)
            # pred_list.append(pred_label)
            #print(pred_label
            #predict_result.append([fid,label])
        # real_list = np.array(real_list)
        # pred_list = np.array(pred_list)
        score = f1_score(real_list,pred_list,average="macro")
        print("宏平均分数",score)
        #predict_result = np.array(predict_result)
        #np.save("predict_result.npy",predict_result)
    # return loss_meter / it_count, f1_meter / it_count

#输出预测结果时还需要加1
if __name__ == "__main__":
    best_f1 = -1
    for epoch in range(0, 1):
        since = time.time()
        #train_loss, train_f1 = train_epoch(model, optimizer, criterion, train_dataloader, show_interval=100)
        #val_loss, val_f1 = val_epoch(model, criterion, val_dataloader)
        test_epoch(model,val_dataloader)
        #print('#epoch:%02d  train_loss:%.3e train_f1:%.3f  val_loss:%0.3e val_f1:%.3f time:%s\n'
        #     % (epoch, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        #torch.save(model.state_dict(), f"shandong_ECGNet_last.mdl")
        #if best_f1 < val_f1:
        #   print(f' f1 imporve from {best_f1} to {val_f1} Save Model...')
        #   torch.save(model.state_dict(), f"shandong_ECGNet_best.mdl")
        #   best_f1 = val_f1

###  val = 0.8746

