import warnings
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import StratifiedKFold
KF = StratifiedKFold(n_splits=5, random_state=2021, shuffle=True)
predicts = []
trainFeatures = []
trainLabels = []
testFeatres = []
from sklearn.metrics import f1_score

with open("train_deep_feature.npy",'rb') as file:
    lgb_feature = np.load(file,allow_pickle=True)
with open("train_feature.npy",'rb') as file:
    lgb_feature_2 = np.load(file,allow_pickle=True)
with open("train_hand_feature.npy",'rb') as file:
    lgb_feature_3 = np.load(file,allow_pickle=True)

# print(lgb_feature.shape)
print(lgb_feature_3.shape)
with open("train_label.npy",'rb') as file:
    lgb_label = np.load(file,allow_pickle=True)

with open("test_deep_feature.npy",'rb') as file:
    test_feature_ = np.load(file,allow_pickle=True)
with open("test_feature.npy",'rb') as file:
    test_feature_2 = np.load(file,allow_pickle=True)
with open("test_hand_feature.npy",'rb') as file:
    test_feature_3 = np.load(file,allow_pickle=True)
# print(test_feature_.shape)
print(test_feature_3.shape)
with open("test_label.npy",'rb') as file:
    test_label_ = np.load(file,allow_pickle=True)
# print(lgb_feature.shape)
# print(lgb_feature_2.shape)
# lgb_feature = np.concatenate((lgb_feature_2,lgb_feature),axis=1)
#
# print(lgb_feature.shape)
#
# test_feature = np.hstack((test_feature_2,test_feature_))

# np.concatenate()
train_feature = []
train_feature_2 = []
train_feature_3 = []
train_label = []
test_feature = []
test_feature_2_ = []
test_feature_3_ = []
test_label = []

for i in range(lgb_feature.shape[0]):
    for item in lgb_feature[i]:
        train_feature.append(item)
for i in range(lgb_feature_2.shape[0]):
    for item in lgb_feature_2[i]:
        train_feature_2.append(item)

for i in range(lgb_label.shape[0]):
    for item in lgb_label[i]:
        train_label.append(item)

for i in range(test_feature_.shape[0]):
    for item in test_feature_[i]:
        test_feature.append(item)
for i in range(test_feature_2.shape[0]):
    for item in test_feature_2[i]:
        test_feature_2_.append(item)
for i in range(test_label_.shape[0]):
    for item in test_label_[i]:
        test_label.append(item)
train_feature  = np.array(train_feature)
train_feature_2  = np.array(train_feature_2)
train_label  = np.array(train_label)
test_feature  = np.array(test_feature)
test_feature_2_  = np.array(test_feature_2_)
test_label  = np.array(test_label)

print(train_feature.shape)
print(train_feature_2.shape)
print(train_label.shape)
print(test_feature.shape)
print(test_feature_2_.shape)
print(test_label.shape)

train_feature = np.hstack((train_feature,train_feature_2))
test_feature = np.hstack((test_feature,test_feature_2_))
print(train_feature.shape)

# lgb训练
predict_result = []
for i in range(12) :
    print("predicting: ",i)
    trn_data = train_feature
    trn_label = train_label
    val_data = test_feature
    val_label = test_label
    oof = np.zeros((2081, ))
    for fold_, (trn_idx, val_idx) in enumerate(KF.split(trn_data, trn_label[:, i])):
        dtrain = lgb.Dataset(trn_data[trn_idx], label=trn_label[:, i][trn_idx])
        dvalid = lgb.Dataset(trn_data[val_idx], label=trn_label[:, i][val_idx])
        params = {
            'objective':'binary',
                        'boosting_type':'gbdt',
                        'metric':'auc',
                        'n_jobs':-1,
                        'learning_rate':0.05,
                        'num_leaves': 2**6,
                        'max_depth':8,
                        'tree_learner':'serial',
                        'colsample_bytree': 0.8,
                        'subsample_freq':1,
                        'subsample':0.8,
                        'num_boost_round':5000,
                        'max_bin':255,
                        'verbose':-1,
                        'seed': 2021,
                        'bagging_seed': 2021,
                        'feature_fraction_seed': 2021,
                        'early_stopping_rounds':300,
            }
        clf = lgb.train(
                        params=params,
                        train_set=dtrain,
                        # num_boost_round=50000,
                        valid_sets=[dvalid],
                        early_stopping_rounds=50,
                        verbose_eval=100,
                        # feval=score_acc2
                    )
        output = clf.predict(val_data, num_iteration=clf.best_iteration)
        oof= oof + clf.predict(val_data, num_iteration=clf.best_iteration)/5
    oof = oof > 0.5
    predict_result.append(oof.astype(int))
result = []
for j in range(val_data.shape[0]):
    item_result = [0]*12
    for i in range(12):
        item_result[i] = predict_result[i][j]
    result.append(item_result)
result = np.array(result)
result = result.reshape(-1,1)
test_label = test_label.reshape(-1,1)
print(f1_score(test_label,result))
# print(test_label[0:10])
# print(result[0:10])

    # clf_lgb.fit(trainFeatures, trainLabels[:,i])
    # predicts.append(clf_lgb.predict(testFeatures).tolist())
