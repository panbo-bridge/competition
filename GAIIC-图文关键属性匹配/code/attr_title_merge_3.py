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

test_data = "./data/contest_data/preliminary_testB.txt"
attr_dict_file = "./code/attr_to_attrvals.json"
out_file = "./data/submission/results_15.txt"

test_pred_title_list = []
test_pred_title_list_2 = []
test_pred_title_list_3 = []

with open("./data/tmp_data/testB_title_visualbert_B3.14fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list.append(data)
with open("./data/tmp_data/testB_title_lxmert.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_2.append(data)
with open("./data/tmp_data/testB_title_uniter_B3.14fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_3.append(data)



# with open("lxmert/new/test_pred_title_uniter.txt", 'r',encoding="utf-8") as f:
#     for i, data in enumerate(tqdm.tqdm(f)):
#         data = json.loads(data)
#         test_pred_title_list_11.append(data)
# with open("lxmert/new/test_pred_title_lxmert.txt", 'r',encoding="utf-8") as f:
#     for i, data in enumerate(tqdm.tqdm(f)):
#         data = json.loads(data)
#         test_pred_title_list_12.append(data)

test_pred_attr_list = []
test_pred_attr_list_2 = []


with open("./data/tmp_data/testB_attr_visualbert_B3.14fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_attr_list.append(data)
with open("./data/tmp_data/testB_attr_lxmert.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_attr_list_2.append(data)
with open("./data/tmp_data/testB_attr_uniter.txt_B3.14fold", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_attr_list_3.append(data)
# test
attr_dict = load_attr_dict(attr_dict_file)
rets = []
error_count_1 = 0
error_count_2 = 0
count_1 = 0
with open(test_data, 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        img_name = data["img_name"]
        query = data["query"]

        item_title = test_pred_title_list[i]
        item_title_2 = test_pred_title_list_2[i]
        item_title_3 = test_pred_title_list_3[i]


        item_attr = test_pred_attr_list[i]
        item_attr_2 = test_pred_attr_list_2[i]
        item_attr_3 = test_pred_attr_list_3[i]
        match = {}


        assert item_title["img_name"] == item_attr["img_name"] == img_name

        for j in range(len(query)):
            if query[j] == "图文":
                title_match_list = []
                title_match_list.append(item_title["match"][query[j]])
                title_match_list.append(item_title_2["match"][query[j]])
                title_match_list.append(item_title_3["match"][query[j]])

                title_count = Counter(title_match_list)
                if title_count[1] >= 2:   #9个里面5的效果最好
                    match[query[j]] = 1
                else:
                    match[query[j]] = 0
            else:
                attr_match_list = []
                attr_match_list.append(item_attr["match"][query[j]])
                attr_match_list.append(item_attr_2["match"][query[j]])
                attr_match_list.append(item_attr_3["match"][query[j]])
                attr_count = Counter(attr_match_list)
                if attr_count[1] >= 2 :
                    match[query[j]] = 1
                else:
                    match[query[j]] = 0

                # match[query[j]] = item_attr_4["match"][query[j]]
                # match[query[j]] = 0
        ret = {
            "img_name": data["img_name"],
            "match":match
        }
        item_chick = ret["match"]

        if item_chick["图文"] == 1:
            count_1 = count_1 + 1
            for key,value in item_chick.items():
                if value != 1:
                    print(item_chick)
                    ret["match"][key] = 1
                    error_count_1 = error_count_1 + 1
            #count_0 = 0
            #for key,value in item_chick.items():
            #    if value == 0:
            #        count_0 = count_0 + 1
            #if count_0 <= 1:
            #    for key,value in item_chick.items():
            #        if value != 1:
            #            print(item_chick)
            #            ret["match"][key] = 1
            #            error_count_1 = error_count_1 + 1
            #else:
            #    ret["match"]["图文"] = 0
            #    print(item_chick)
            #    error_count_2 = error_count_2 + 1

        rets.append(json.dumps(ret, ensure_ascii=False)+'\n')
print(error_count_1,error_count_2)
print("标题匹配的个数：",count_1)
with open(out_file, 'w',encoding="utf-8") as f:
    f.writelines(rets)
