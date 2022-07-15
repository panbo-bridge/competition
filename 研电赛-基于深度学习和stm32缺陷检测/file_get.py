import os
import re
import time
path = "./dataset" #文件夹目录
files= os.listdir(path) #得到文件夹下的所有文件名称

# 优化格式化化版本
# time_now = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
time_now = '20210522111215'
# s = []
# for file in files: #遍历文件夹
#     if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
#         f = open(path+"/"+file); #打开文件
#         iter_f = iter(f); #创建迭代器
#         str = ""
#         for line in iter_f: #遍历文件，一行行遍历，读取文本
#             str = str + line
#         s.append(str) #每个文件的文本存到list中
for i in files:
    t=re.match(time_now, i)  # 在起始位置匹配
    if t:
        print(i)
# print(re.match('com', 'www.runoob.com'))   
# print(files) #打印结果