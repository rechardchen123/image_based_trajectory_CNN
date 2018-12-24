#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/24/2018 2:28 AM 
# @Author : Xiang Chen (Richard)
# @File : test_image_data_processing.py 
# @Software: PyCharm
import tensorflow as tf
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def int_to_one_hot(labels):
    label = []
    if labels[1] == -1:
        label.append([0, 0, 0])
    else:
        label.append([1, 0, 0])
    if labels[2] == -1:
        label.append([0, 0, 0])
    else:
        label.append([0, 1, 0])
    if labels[3] == -1:
        label.append([0, 0, 0])
    else:
        label.append([0, 0, 1])
    return label

label_dir  = r'C:\Users\LPT-ucesxc0\AIS-Data\test_image_process'
file_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\test_image_process\AIS_trajectory_image_clip_labels'
labels = []
static_state = []
static_label = []
normal_navigation = []
normal_label = []
maneuvring = []
maneuvring_label = []
with open (label_dir +r'\label.txt') as f:
        for line in f.readlines():
            labels.append(line)

for i in range(len(labels)):
    item = labels[i].replace('\n','')
    item = labels[i].split(',')
    for j in range(len(item)):
        if item[1]== '1' and item[2]== '-1' and item[3]=='-1':
            static_label.append(int_to_one_hot(item))
            image_name_static = item[0]
            image_path_static = os.path.join(file_dir,image_name_static)
            if os.path.isfile(image_path_static):
                static_image = Image.open(image_path_static)
                static_image = static_image.tobytes()
                static_state.append(static_image)
        elif item[2]== '1' and item[3]== '-1':
            normal_label.append(int_to_one_hot(item))
            image_name_normal_navigation = item[0]
            image_path_normal_navigation = os.path.join(file_dir,image_path_normal_navigation)
            if os.path.isfile(image_path_normal_navigation):
                normal_navigation_image = Image.open(image_path_normal_navigation)
                normal_navigation.append(normal_navigation_image)
        elif item[3]== '1':
            maneuvring_label.append(int_to_one_hot(item))
            image_name_maneuvring = item[0]
            image_path_maneuvring = os.path.join(file_dir,image_name_maneuvring)
            if os.path.isfile(image_path_maneuvring):
                maneuvring_image = Image.open(image_path_maneuvring)
                maneuvring.append(maneuvring_image)



