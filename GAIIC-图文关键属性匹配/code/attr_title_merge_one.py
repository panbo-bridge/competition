import os, sys
import re
import json
import torch
import numpy as np
import itertools
import tqdm
from collections import Counter
def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r',encoding="utf-8") as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict
def match_attrval(title, attr, attr_dict):
    # 在title中匹配属性值
    attrvals = "|".join(attr_dict[attr])
    ret = re.findall(attrvals, title)
    return "{}{}".format(attr, ''.join(ret))

test_data = "/home/mw/input/track1_contest_4362/semi_testA.txt"
attr_dict_file = "./code/attr_to_attrvals.json"
out_file = "./data/submission/results.txt"

test_pred_title_list = []
with open("./data/tmp_data/testB_title_visualbert_3fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list.append(data)
test_pred_attr_list = []
with open("./data/tmp_data/testB_attr_visualbert.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_attr_list.append(data)

attr_dict = load_attr_dict(attr_dict_file)
rets = []
error_count_1 = 0
error_count_2 = 0
img_title_true = 0
with open(test_data, 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        img_name = data["img_name"]
        query = data["query"]

        item_title = test_pred_title_list[i]
        item_attr = test_pred_attr_list[i]
        match = {}
        assert item_title["img_name"] == item_attr["img_name"] == img_name

        for j in range(len(query)):
            if query[j] == "图文":

                match[query[j]] = item_title["match"][query[j]]
                # match[query[j]] = 0


            else:
                match[query[j]] = item_attr["match"][query[j]]
        ret = {
            "img_name": data["img_name"],
            "match":match
        }
        item_chick = ret["match"]

        if item_chick["图文"] == 1:
            img_title_true = img_title_true + 1
            # for key,value in item_chick.items():
            #     if value != 1:
            #         print(item_chick)
            #         ret["match"][key] = 1
            #         error_count_1 = error_count_1 + 1

        rets.append(json.dumps(ret, ensure_ascii=False)+'\n')
print("图和标题匹配的个数：",img_title_true)
with open(out_file, 'w',encoding="utf-8") as f:
    f.writelines(rets)
