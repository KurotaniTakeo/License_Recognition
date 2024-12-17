"""
@Author: HuKai
@Date: 2022/5/29  10:44
@github: https://github.com/HuKai97
"""
import os
import random

import shutil
from shutil import copy2

trainfiles = os.listdir(r"D:\CCPD2019\CCPD2019\ccpd_base")  #（图片文件夹）
num_train = 10000
print("num_train: " + str(num_train) )
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)  # 打乱顺序
num = 0
trainDir = r"D:\CCPD2019\train"   #（将图片文件夹中的6份放在这个文件夹下）
validDir = r"D:\CCPD2019\val"     #（将图片文件夹中的2份放在这个文件夹下）
detectDir = r"D:\CCPD2019\test"   #（将图片文件夹中的2份放在这个文件夹下）
for i in index_list:
    fileName = os.path.join(r"D:\CCPD2019\CCPD2019\ccpd_base", trainfiles[i])  #（图片文件夹）+图片名=图片地址
    if num < num_train*0.7:  # 7:1:2
        print(str(fileName))
        copy2(fileName, trainDir)
    elif num < num_train*0.8:
        print(str(fileName))
        copy2(fileName, validDir)
    else:
        print(str(fileName))
        copy2(fileName, detectDir)
    num += 1