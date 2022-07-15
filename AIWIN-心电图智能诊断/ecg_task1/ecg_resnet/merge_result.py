import pandas as pd
import os
import glob
import numpy as np
train_df_1 = pd.read_csv('answer_1.csv')
train_df_2 = pd.read_csv('answer_2.csv')
train_df_3 = pd.read_csv('answer_3.csv')
# train_df_4 = pd.read_csv('answer_4.csv')
# train_df_5 = pd.read_csv('answer_5.csv')
# train_df['tag'] = train_df['tag'].astype(np.float32)
test_pred = []
for i in range(400):
    res = []
    value1 = train_df_1["tag"][i]
    value2 = train_df_2["tag"][i]
    value3 = train_df_3["tag"][i]
    # value4 = train_df_4["tag"][i]
    # value5 = train_df_5["tag"][i]
    res.append(value1)
    res.append(value2)
    res.append(value3)
    # res.append(value4)
    # res.append(value5)
    if res.count(1) >=1:
        result = 1
    else:
        result = 0
    test_pred.append(result)
test_path = glob.glob('./val/*.mat')
test_path = [os.path.basename(x)[:-4] for x in test_path]
print("a")
test_path.sort()
test_answer = pd.DataFrame({
    'name': test_path,
    'tag': test_pred
}).to_csv('answer.csv', index=None)
