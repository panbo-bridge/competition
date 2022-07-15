# -*- coding: utf-8 -*-
'''
@time: 2019/9/12 15:16

@ author: javis
'''
import torch
import numpy as np
import time,os
from sklearn.metrics import f1_score
from torch import nn
pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8,pred_9,pred_10,pred_11,pred_12 = [],[],[],[],[],[],[],[],[],[],[],[],
predict_result = []
real_result = []
def calc_threshild(y_true, y_pre):
    y_true = y_true.cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.cpu().detach().numpy()
    for j in range(y_true.shape[0]):
        if y_true[j][0] == 1:
            pred_1.append(y_pre[j][0])
        if y_true[j][1] == 1:
            pred_2.append(y_pre[j][1])
        if y_true[j][2] == 1:
            pred_3.append(y_pre[j][2])
        if y_true[j][3] == 1:
            pred_4.append(y_pre[j][3])
        if y_true[j][4] == 1:
            pred_5.append(y_pre[j][4])
        if y_true[j][5] == 1:
            pred_6.append(y_pre[j][5])
        if y_true[j][6] == 1:
            pred_7.append(y_pre[j][6])
        if y_true[j][7] == 1:
            pred_8.append(y_pre[j][7])
        if y_true[j][8] == 1:
            pred_9.append(y_pre[j][8])
        if y_true[j][9] == 1:
            pred_10.append(y_pre[j][9])
        if y_true[j][10] == 1:
            pred_11.append(y_pre[j][10])
        if y_true[j][11] == 1:
            pred_12.append(y_pre[j][11])
    print("--------------------")
    print(calc_mean(pred_1))
    print(calc_mean(pred_2))
    print(calc_mean(pred_3))
    print(calc_mean(pred_4))
    print(calc_mean(pred_5))
    print(calc_mean(pred_6))
    print(calc_mean(pred_7))
    print(calc_mean(pred_8))
    print(calc_mean(pred_9))
    print(calc_mean(pred_10))
    print(calc_mean(pred_11))
    print(calc_mean(pred_12))
    print("********************")
def calc_mean(label_list):
    sum_all = 0
    leng = len(label_list)
    for i in range(leng):
        sum_all = sum_all + label_list[i]
    if leng != 0:
        return sum_all / leng
    else:
        return 0
def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

#计算F1score
def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    # print(y_pre)
    for item in y_pre:
        predict_result.append(item)
    for item in y_true:
        real_result.append(item)
    save_true = real_result
    save_pre = predict_result
    #
    # real_result = np.array(real_result).reshape(-1,1)
    # predict_result = np.array(predict_result).reshape(-1,1)
    np.save("save_true.npy",save_true)
    np.save("save_pre.py",save_pre)

    return f1_score(y_true, y_pre)


#打印时间
def print_time_cost(since):
    time_elapsed = time.time() - since
    return '{:.0f}m{:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)


# 调整学习率
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

#多标签使用类别权重
class WeightedMultilabel(nn.Module):
    def __init__(self, weights: torch.Tensor):
        super(WeightedMultilabel, self).__init__()
        self.cerition = nn.BCEWithLogitsLoss(reduction='none')
        self.weights = weights

    def forward(self, outputs, targets):
        loss = self.cerition(outputs, targets)
        return (loss * self.weights).mean()

