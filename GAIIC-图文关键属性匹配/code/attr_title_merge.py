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
out_file = "./data/submission/results.txt"

test_pred_title_list = []
test_pred_title_list_2 = []
test_pred_title_list_3 = []
test_pred_title_list_4 = []
test_pred_title_list_5 = []
test_pred_title_list_6 = []
test_pred_title_list_7 = []
test_pred_title_list_8 = []
test_pred_title_list_9 = []
# test_pred_title_list_10 = []
# test_pred_title_list_11 = []
# test_pred_title_list_12 = []
# test_pred_title_list_13 = []
# test_pred_title_list_14 = []
# test_pred_title_list_15 = []
# test_pred_title_list_16 = []
# test_pred_title_list_17 = []
# test_pred_title_list_18 = []




with open("./data/tmp_data/testB_title_visualbert_1fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list.append(data)
with open("./data/tmp_data/testB_title_visualbert_2fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_2.append(data)
with open("./data/tmp_data/testB_title_visualbert_3fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_3.append(data)
with open("./data/tmp_data/testB_title_visualbert_4fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_4.append(data)
with open("./data/tmp_data/testB_title_visualbert_5fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_5.append(data)
with open("./data/tmp_data/testB_title_visualbert_6fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_6.append(data)
with open("./data/tmp_data/testB_title_visualbert_7fold.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_7.append(data)
with open("./data/tmp_data/testB_title_lxmert.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_8.append(data)
with open("./data/tmp_data/testB_title_uniter.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_title_list_9.append(data)



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
test_pred_attr_list_3 = []
test_pred_attr_list_4 = []
test_pred_attr_list_5 = []
test_pred_attr_list_6 = []
test_pred_attr_list_7 = []
test_pred_attr_list_8 = []
test_pred_attr_list_9 = []


# with open("lxmert/new/test_pred_attr_visualbert_1fold_score.txt", 'r',encoding="utf-8") as f:
#     for i, data in enumerate(tqdm.tqdm(f)):
#         data = json.loads(data)
#         test_pred_attr_list.append(data)
# with open("lxmert/new/test_pred_attr_visualbert_2fold_score.txt", 'r',encoding="utf-8") as f:
#     for i, data in enumerate(tqdm.tqdm(f)):
#         data = json.loads(data)
#         test_pred_attr_list_2.append(data)
# with open("lxmert/new/test_pred_attr_visualbert_3fold_score.txt", 'r',encoding="utf-8") as f:
#     for i, data in enumerate(tqdm.tqdm(f)):
#         data = json.loads(data)
#         test_pred_attr_list_3.append(data)
# with open("lxmert/new/test_pred_attr_visualbert_4fold_score.txt", 'r',encoding="utf-8") as f:
#     for i, data in enumerate(tqdm.tqdm(f)):
#         data = json.loads(data)
#         test_pred_attr_list_4.append(data)
# with open("lxmert/new/test_pred_attr_visualbert_5fold_score.txt", 'r',encoding="utf-8") as f:
#     for i, data in enumerate(tqdm.tqdm(f)):
#         data = json.loads(data)
#         test_pred_attr_list_5.append(data)
# with open("lxmert/new/test_pred_attr_visualbert_6fold_score.txt", 'r',encoding="utf-8") as f:
#     for i, data in enumerate(tqdm.tqdm(f)):
#         data = json.loads(data)
#         test_pred_attr_list_6.append(data)

with open("./data/tmp_data/testB_attr_visualbert.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_attr_list.append(data)
with open("./data/tmp_data/testB_attr_lxmert.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_attr_list_2.append(data)
with open("./data/tmp_data/testB_attr_uniter.txt", 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        test_pred_attr_list_3.append(data)
# test
attr_dict = load_attr_dict(attr_dict_file)
rets = []
error_count_1 = 0
error_count_2 = 0
with open(test_data, 'r',encoding="utf-8") as f:
    for i, data in enumerate(tqdm.tqdm(f)):
        data = json.loads(data)
        img_name = data["img_name"]
        query = data["query"]

        item_title = test_pred_title_list[i]
        item_title_2 = test_pred_title_list_2[i]
        item_title_3 = test_pred_title_list_3[i]
        item_title_4 = test_pred_title_list_4[i]
        item_title_5 = test_pred_title_list_5[i]
        item_title_6 = test_pred_title_list_6[i]
        item_title_7 = test_pred_title_list_7[i]
        item_title_8 = test_pred_title_list_8[i]
        item_title_9 = test_pred_title_list_9[i]



        item_attr = test_pred_attr_list[i]
        item_attr_2 = test_pred_attr_list_2[i]
        item_attr_3 = test_pred_attr_list_3[i]
        # item_attr_4 = test_pred_attr_list_4[i]
        # item_attr_5 = test_pred_attr_list_5[i]
        # item_attr_6 = test_pred_attr_list_6[i]
        # item_attr_7 = test_pred_attr_list_7[i]
        # item_attr_8 = test_pred_attr_list_8[i]
        # item_attr_9 = test_pred_attr_list_9[i]
        match = {}


        assert item_title["img_name"] == item_attr["img_name"] == img_name

        for j in range(len(query)):
            if query[j] == "图文":
                # title_p_visualbert = item_title["match"][query[j]] + item_title_2["match"][query[j]] + item_title_3["match"][query[j]] + item_title_4["match"][query[j]] + item_title_5["match"][query[j]] + item_title_6["match"][query[j]] + item_title_7["match"][query[j]] + item_title_8["match"][query[j]] + item_title_9["match"][query[j]] + item_title_10["match"][query[j]]
                # title_p_uniter = item_title_11["match"][query[j]] + item_title_12["match"][query[j]] + item_title_13["match"][query[j]]
                # # title_p = item_title_2["match"][query[j]]
                # title_p_visualbert = title_p_visualbert / 10.0
                # title_p_uniter = title_p_uniter / 3.0
                # title_p_lxmert = item_title_14["match"][query[j]]
                #
                # title_p = 0.6*title_p_visualbert + 0.4*title_p_uniter
                # if title_p >= 0.65:        #目前uniter阀值是0.7，visualbert阀值是0.5
                #     match[query[j]] = 1
                # else:
                #     match[query[j]] = 0

                title_match_list = []
                title_match_list.append(item_title["match"][query[j]])
                title_match_list.append(item_title_2["match"][query[j]])
                title_match_list.append(item_title_3["match"][query[j]])
                title_match_list.append(item_title_4["match"][query[j]])
                title_match_list.append(item_title_5["match"][query[j]])
                title_match_list.append(item_title_6["match"][query[j]])
                title_match_list.append(item_title_7["match"][query[j]])
                title_match_list.append(item_title_8["match"][query[j]])
                title_match_list.append(item_title_9["match"][query[j]])

                title_count = Counter(title_match_list)
                if title_count[1] > 5:   #9个里面5的效果最好
                    match[query[j]] = 1
                else:
                    match[query[j]] = 0

                # match[query[j]] = item_title_9["match"][query[j]]
                # match[query[j]] = 0


            else:

                # attr_p = item_attr["match"][query[j]] + item_attr_2["match"][query[j]] + item_attr_3["match"][query[j]] + item_attr_4["match"][query[j]] + item_attr_5["match"][query[j]] + item_attr_6["match"][query[j]] + item_attr_7["match"][query[j]]
                # attr_p = attr_p / 7.00
                # if attr_p >= 0.45:
                #     match[query[j]] = 1
                # else:
                #     match[query[j]] = 0
                attr_match_list = []
                attr_match_list.append(item_attr["match"][query[j]])
                attr_match_list.append(item_attr_2["match"][query[j]])
                attr_match_list.append(item_attr_3["match"][query[j]])
                # attr_match_list.append(item_attr_4["match"][query[j]])
                # attr_match_list.append(item_attr_5["match"][query[j]])
                # attr_match_list.append(item_attr_6["match"][query[j]])
                # attr_match_list.append(item_attr_7["match"][query[j]])
                # attr_match_list.append(item_attr_8["match"][query[j]])
                # attr_match_list.append(item_attr_9["match"][query[j]])
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
with open(out_file, 'w',encoding="utf-8") as f:
    f.writelines(rets)
