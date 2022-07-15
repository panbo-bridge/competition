import os, sys
import re
import json
import torch
import numpy as np
import itertools
import tqdm
test_data_1 = "./testA.txt"
test_data_2 = "./testB.txt"
catch_dict = {}
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
    if len(ret) != 0:
        return ret[0]
    else:
        return None
    # print(ret)
    # return "{}{}".format(attr, ''.join(ret))
attr_dict_file = "./attr_to_attrvals.json"
attr_dict = load_attr_dict(attr_dict_file)
keys = list(attr_dict.keys())
with open(test_data_1, 'r',encoding="utf-8") as f:
    for i,data in enumerate(tqdm.tqdm(f)):
        item = json.loads(data)
        title = item["title"]
        for a in keys:
            value = match_attrval(title, a, attr_dict)
            if value != None:
                if value in catch_dict.keys():
                    catch_dict[value] = catch_dict[value] + 1
                else:
                    catch_dict[value] = 1
catch_dict = sorted(catch_dict.items(), key=lambda x: x[1])
print(catch_dict)
catch_dict = {}
with open(test_data_2, 'r',encoding="utf-8") as f:
    for i,data in enumerate(tqdm.tqdm(f)):
        item = json.loads(data)
        title = item["title"]
        for a in keys:
            value = match_attrval(title, a, attr_dict)
            if value != None:
                if value in catch_dict.keys():
                    catch_dict[value] = catch_dict[value] + 1
                else:
                    catch_dict[value] = 1
catch_dict = sorted(catch_dict.items(), key=lambda x: x[1])
print(catch_dict)