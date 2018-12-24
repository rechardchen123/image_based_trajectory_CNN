#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 12/21/2018 3:58 PM
# @Author : Xiang Chen (Richard)
# @File : training_model.py
# @Software: VS code
import os
import numpy as np
import tensorflow as tf
import new_image_data_preprocessing


#define the variables
N_CLASSES = 3  # static,normal_navigation,maneuvring
BATCH_SIZE = 20
CAPACITY = 200
MAX_STEP = 200
learning_rate = 0.0001

#get the batch
train_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\test_image_process\AIS_trajectory_image_clip_labels'
label_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\test_image_process'
logs_train_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\test_image_process'

#get the train and test data and labels
train, train_label, test, test_label = new_image_data_preprocessing.get_files(train_dir,label_dir, 0.3)
print(train,train_label,test,test_label)

