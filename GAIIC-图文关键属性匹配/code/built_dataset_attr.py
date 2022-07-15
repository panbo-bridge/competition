import json
import itertools
import tqdm
import re
import pandas as pd
# attr_file_path = r"C:\Users\pb\Desktop\GAIIC_track1_baseline\GAIIC_track1_baseline\src\data\attr_to_attrvals.json"  #属性词表的路径
# input_filename_1 = r"C:\Users\pb\Desktop\GAIIC_track1_baseline\GAIIC_track1_baseline\src\data\train_coarse.txt"
# input_filename_2 = r"C:\Users\pb\Desktop\GAIIC_track1_baseline\GAIIC_track1_baseline\src\data\train_fine.txt"
input_filename_1 = r"./data/contest_data/train_coarse.txt"
input_filename_2 = r"./data/contest_data/train_fine.txt"
attr_file_path = r"./code/attr_to_attrvals.json"

input_filename = [input_filename_1,input_filename_2]

attr_data_save_path = "./data/tmp_data/attr_dataset/" #该属性的feature存储位置
attr_label_save_path = '/data/tmp_data/attr_label.csv'  #该属性的label存储位置

def load_attr_dict(file):
    # 读取属性字典
    with open(file, 'r',encoding="utf-8") as f:
        attr_dict = {}
        for attr, attrval_list in json.load(f).items():
            attrval_list = list(map(lambda x: x.split('='), attrval_list))
            attr_dict[attr] = list(itertools.chain.from_iterable(attrval_list))
    return attr_dict
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
def match_attrval(title, attr, attr_dict):
    # 在title中匹配属性值
    attrvals = "|".join(attr_dict[attr])
    ret = re.findall(attrvals, title)
    if len(ret) == 0:
        return None
    else:
        #return "{}{}".format(attr, ''.join(ret[0]))
        return ret[0]
    # else:
    #     label_flag = False
    #     for item in ret:
    #         label = attr_label_dict[item]
    #         if label_flag and label != label_flag:
    #             print("----------------------------------------------------------------------------------------------------------------")
    #             print("存在坏的title",title)
    #             print("----------------------------------------------------------------------------------------------------------------")
    #             return None
    #         else:
    #             label_flag = label
    #     return ret[0]
def save_lable_to_csv(img_name_list,img_label_list):
    #保存feature在该属性下的label
    assert len(img_label_list) == len(img_label_list)
    pd.DataFrame({
        'name': img_name_list,
        'tag': img_label_list
    }).to_csv(attr_label_save_path, index=None)
def read_feature(input_filename):
    #从txt文件中读取图片的标题，feature，匹配等信息
    attr_dict = load_attr_dict(attr_file_path)
    keys = list(attr_dict.keys())
    img_name_list = []
    img_label_list = []
    sample_count = 0
    for file in input_filename:
        with open(file, 'r',encoding="utf-8") as f:
            for line in tqdm.tqdm(f):
                item = json.loads(line)
                if item['match']['图文']:
                    img_name = item["img_name"]
                    title = item["title"]
                    for i,attr in enumerate(keys):
                        attr_name = match_attrval(title,attr,attr_dict)
                        if attr_name:
                            label = 1
                            img_name_ = img_name + "_" + str(i)
                            print(attr_name,img_name_)
                            img_name_list.append(img_name_)
                            img_label_list.append(label)
                            item["attr"] = attr_name
                            sample_count = sample_count +1
                            save_feature_to_json(img_name_,item)
    print(sample_count)
    save_lable_to_csv(img_name_list,img_label_list)
if __name__ == "__main__":
    read_feature(input_filename)
