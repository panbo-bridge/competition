import json
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from pandas.core.frame import DataFrame
from param import args
import itertools
import tqdm
import re
import random
import jieba
sub_count = 0
def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r',encoding="utf-8") as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict
attr_file_path = args.attr_file_path
attr_dict = load_attr_dict(attr_file_path)
keys = list(attr_dict.keys())
def match_attrval(title, attr, attr_dict):
    # 在title中匹配属性值
    attrvals = "|".join(attr_dict[attr])
    ret = re.findall(attrvals, title)
    return ret
def shuffle_title(title):
    #查找要替换的字符串
    catch_attr = []
    title_ex = title
    for attr in keys:
        ret = match_attrval(title,attr,attr_dict)
        if ret != [] and ret[0] not in catch_attr:
            catch_attr.append(ret[0])
            title_ex = title_ex.replace(ret[0],"")
    words = jieba.cut(title_ex)
    words_list = []
    for word in words:
        words_list.append(word)
    all_words = words_list + catch_attr
    random.shuffle(all_words)
    title = ''.join(all_words)
    return title
def negative(title,target):
    sub_attr = {}
    #查找要替换的字符串

    #if "拉链" in title and "鞋" not in "title" and "裤" not in "title":
    #    print(title,"---------------------------")
    catch_attr = set()
    for attr in keys:
        ret = match_attrval(title,attr,attr_dict)
        #print(ret)
        if ret != []:#attr的话是全部替换，title是以50%的概率替换
            #catch_attr.add(ret[0])
            sub_attr[attr] = ret[0]
    if "鞋" in title and "裤门襟" in sub_attr.keys():
        sub_attr.pop("裤门襟")
    elif "鞋" not in title and "闭合方式" in sub_attr.keys():
        sub_attr.pop("闭合方式")
    if "裤" not in title and "裤门襟" in sub_attr.keys():
        sub_attr.pop("裤门襟")
    new_sub_attr = {}
    if len(sub_attr) <= 1:
        new_sub_attr = sub_attr
    else:
        for k,v in sub_attr.items():
            if random.randint(0,1):
                new_sub_attr[k] = v
    #z整个数据集中属性最多的是五个
    #print(new_sub_attr)
    if new_sub_attr != {}:
        with open(attr_file_path, 'r',encoding="utf-8") as f:
            attr_dict_old = json.load(f)
        for key,value in new_sub_attr.items():
            attrval_list = attr_dict_old[key]
            #查找到要替换的子串，删除构建可以用于替换的数据集
            for item in attrval_list:
                if value in item:
                    attrval_list.remove(item)
                    break
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attrval_list = list(itertools.chain.from_iterable(attrval_list))
            sub_value = random.choice(attrval_list)
            title = title.replace(value,sub_value)
            global sub_count
            sub_count = sub_count + 1
        #print(title)
        return title,0
    else:
        return title,target

def negative_attr(title,target):
    sub_attr = {}
    #查找要替换的字符串
    catch_attr = set()
    for attr in keys:
        ret = match_attrval(title,attr,attr_dict)
        #print(ret)
        if ret != []:#attr的话是全部替换，title是以50%的概率替换
            #catch_attr.add(ret[0])
            sub_attr[attr] = ret[0]
    #if ("鞋" in title or "裤" not in title) and "裤门襟" in sub_attr.keys():
    #    sub_attr.pop("裤门襟")
    #elif "鞋" not in title and "闭合方式" in sub_attr.keys():
    #    sub_attr.pop("闭合方式")
    #z整个数据集中属性最多的是五个
    #print(new_sub_attr)
    new_sub_attr = {}
    if len(sub_attr) <= 1:
        new_sub_attr = sub_attr
    else:
        #print(sub_attr)
        sub_key = list(sub_attr.keys())
        c_key = random.choice(sub_key)
        new_sub_attr[c_key] = sub_attr[c_key]
        #print(new_sub_attr)
    if new_sub_attr != {}:
        with open(attr_file_path, 'r',encoding="utf-8") as f:
            attr_dict_old = json.load(f)
        for key,value in new_sub_attr.items():
            attrval_list = attr_dict_old[key]
            #查找到要替换的子串，删除构建可以用于替换的数据集
            for item in attrval_list:
                if value in item:
                    attrval_list.remove(item)
                    break
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attrval_list = list(itertools.chain.from_iterable(attrval_list))
            sub_value = random.choice(attrval_list)
            title = title.replace(value,sub_value)
            global sub_count
            sub_count = sub_count + 1
        return title,0
    else:
        return title,target


def syn_sub(title):
    #同义替换
    sub_attr = {}
    #查找要替换的字符串
    catch_attr = set()
    title_list = []
    #title_list.append(title)
    for attr in keys:
        ret = match_attrval(title,attr,attr_dict)
        #print(ret)
        if ret != [] and ret[0] not in catch_attr:#attr的话是全部替换，title是以50%的概率替换
            catch_attr.add(ret[0])
            sub_attr[attr] = ret[0]
    if sub_attr != {}:
        with open(attr_file_path, 'r',encoding="utf-8") as f:
            attr_dict_old = json.load(f)
        for key,value in sub_attr.items():
            attrval_list = attr_dict_old[key]
            #查找到要替换的子串，删除构建可以用于替换的数据集
            sub = None
            for item in attrval_list:
                if "=" in item:
                    sub_list = item.split("=")
                    if value in sub_list:
                        sub = sub_list
                        break
            #print(sub_list,value)
            #sub_list.remove(value)
            #print(title)
            if sub:
                sub_value = random.choice(sub_list)
            #随机替换和插入相等属性
                title = title.replace(value,sub_value)
                title_list.append(title)
    return title_list
def syn_sub_or_add(title,target):
    #同义替换
    sub_attr = {}
    #查找要替换的字符串
    catch_attr = set()
    for attr in keys:
        ret = match_attrval(title,attr,attr_dict)
        #print(ret)
        if ret != []:#attr的话是全部替换，title是以50%的概率替换
            #catch_attr.add(ret[0])
            sub_attr[attr] = ret[0]
    #print(sub_attr)
    if sub_attr != {}:
        with open(attr_file_path, 'r',encoding="utf-8") as f:
            attr_dict_old = json.load(f)
        for key,value in sub_attr.items():
            attrval_list = attr_dict_old[key]
            #查找到要替换的子串，删除构建可以用于替换的数据集
            sub = None
            for item in attrval_list:
                if "=" in item:
                    sub_list = item.split("=")
                    if value in sub_list:
                        #print(sub_list)
                        sub = sub_list
                        break
            #print(title)
            if sub:
                sub_value = random.choice(sub)
            #随机替换和插入相等属性
                title = title.replace(value,sub_value)
    return title,target

def hide_attr(title,target):
    #隐藏属性
    sub_attr = {}
    #查找要替换的字符串
    catch_attr = set()
    for attr in keys:
        ret = match_attrval(title,attr,attr_dict)
        #print(ret)
        if ret != [] and ret[0] not in catch_attr:#attr的话是全部替换，title是以50%的概率替换
            catch_attr.add(ret[0])
            sub_attr[attr] = ret[0]
    if sub_attr != {}:
        with open(attr_file_path, 'r',encoding="utf-8") as f:
            attr_dict_old = json.load(f)
        for key,value in sub_attr.items():
            attrval_list = attr_dict_old[key]
            #查找到要替换的子串，删除构建可以用于替换的数据集
            for item in attrval_list:
                if value in item:
                    break
            if random.randint(0,3) == 2:
                title = title.replace(value,"")
    return title,target
class JsonDataset(Dataset):
    """
    A generic data loader where the samples are arranged in this way:
    dd = {'train': train, 'val': val, "idx2name": idx2name, 'file2idx': file2idx}
    """
    def __init__(self, train=True):
        super(JsonDataset, self).__init__()
        dd = torch.load(args.train_pth_path)
        self.label_num = dd["label_num"]
        self.train = train
        if train:
            self.data = dd["train"]
        else:
            self.data = dd["val"]
        self.file2idx = dd['file2idx']
    def read_feature_from_json(self,img_name):
        #从保存的json中读取feature
        with open(img_name) as f:
            item = json.load(f)
        feature = item["feature"]
        #title
        #title = item["title"] #图文匹配填title,属性匹配填attr
        #return feature,title
        if args.mode == "title":
            title = item["title"]
        else:
            title = item["attr"]
        #title = item["title"]
        #if "鞋" in title and attr in ["系带","松紧带","拉链"]:
        #    attr_title = "鞋" + attr
        #    print(title,attr_title)
        #else:
        #    attr_title = attr
        #    if attr in ["系带","拉链","松紧"]:
        #        print(title,attr_title)
        return feature,title
    def __getitem__(self, index):
        fid = self.data[index]
        file_path = args.dataset_path +  fid
        feature,title = self.read_feature_from_json(file_path)
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0)
        target = self.file2idx[fid]
        label = target[0]
        #print(title,label)
        #隐藏部分属性
        #if label == 1 and random.randint(0,1):
        #    title,label = hide_attr(title,label)
        #-------------------------------------------------------对测试集的数据也进行数据增强了，title的话是不应该的
        #构造负样本属性预测不管测试集还是训练集都应该生成负样本  self.train
        if args.mode == "title":
            if self.train and label == 1 and random.randint(0,1):
                title,label = negative(title,label)
        else:
            if label == 1 and random.randint(0,1):
                title,label = negative_attr(title,label)
        #替换同义词或增加一个同义词
        if self.train and random.randint(0,1):
            title,label = syn_sub_or_add(title,label)
        #打乱词序属性不用，title需要
        if args.mode == "title" and self.train and random.randint(0,1):
            title = shuffle_title(title)
        #print(title,label)
        target = [label]
        target = torch.tensor(target, dtype=torch.float32)
        return feature,title,target
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    data = JsonDataset()
    for i in range(20):
        data[i]
