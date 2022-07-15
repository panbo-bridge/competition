import numpy as np
from sklearn.metrics import f1_score
with open("save_true.npy",'rb') as file:
    y_true = np.load(file,allow_pickle=True)
with open("save_pre.py.npy",'rb') as file:
    y_pre = np.load(file,allow_pickle=True)
# y_pre = y_pre.reshape(-1,1)
# y_true = y_true.reshape(-1,1)
print(y_pre)
print(f1_score(y_true,y_pre))

#原始0.8752
#lgb后0.8822
#五折后0.8849
#五折深度特征0.8807
#概率特征 + 深度特征 = 0.8876