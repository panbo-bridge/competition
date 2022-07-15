import json
import itertools
import tqdm
import re
import pandas as pd
#input_filename_1 = r"C:\Users\pb\Desktop\GAIIC_track1_baseline\GAIIC_track1_baseline\src\data\train_coarse.txt"
#input_filename_2 = r"C:\Users\pb\Desktop\GAIIC_track1_baseline\GAIIC_track1_baseline\src\data\train_fine.txt"
# input_filename_1 = r"/home/panbo/img-text_matching/GAIIC_track1_baseline/src/data/train_coarse.txt"
# input_filename_2 = r"/home/panbo/img-text_matching/GAIIC_track1_baseline/src/data/train_fine.txt"
input_filename_1 = r"./data/contest_data/train_coarse.txt"
input_filename_2 = r"./data/contest_data/train_fine.txt"
input_filename = [input_filename_1,input_filename_2]

attr_data_save_path = "./data/tmp_data/title_dataset/" #该属性的feature存储位置
attr_label_save_path = './data/tmp_data/title_label.csv'  #该属性的label存储位置

def save_feature_to_json(img_name,feature):
    #保存feature为json
    img_name = attr_data_save_path + img_name
    with open(img_name, 'w') as f:
        json.dump(feature, f)
def read_feture_from_json(img_name):
    #从保存的json中读取feature
    img_name = attr_data_save_path + img_name
    with open(img_name) as f:
        feature = json.load(f)
    return feature

def save_lable_to_csv(img_name_list,img_label_list):
    #保存feature在该属性下的label
    assert len(img_label_list) == len(img_label_list)
    pd.DataFrame({
        'name': img_name_list,
        'tag': img_label_list
    }).to_csv(attr_label_save_path, index=None)
def read_feature(input_filename):
    #从txt文件中读取图片的标题，feature，匹配等信息
    img_name_list = []
    img_label_list = []
    max_len = 0
    for file in input_filename:
        with open(file, 'r',encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                item = json.loads(line)
                item["title"] = re.sub("\d+年","",item["title"])
                title = item["title"]
                if len(title) > max_len:
                    max_len = len(title)
                    print(title)
                if item['match']['图文']:
                    img_name = item["img_name"]
                    label = 1
                    img_name_list.append(img_name)
                    img_label_list.append(label)
                    save_feature_to_json(img_name,item)
                else:
                    img_name = item["img_name"]
                    label = 0
                    img_name_list.append(img_name)
                    img_label_list.append(label)
                    save_feature_to_json(img_name,item)
    print(max_len)
    save_lable_to_csv(img_name_list,img_label_list)
if __name__ == "__main__":
    read_feature(input_filename)
#最长文本长度为36
