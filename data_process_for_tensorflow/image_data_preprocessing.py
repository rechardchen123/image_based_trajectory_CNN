#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/16/2018 6:48 PM 
# @Author : Xiang Chen (Richard)
# @File : image_data_preprocessing.py 
# @Software: PyCharm
import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

"""
This is a method for preprocessing the image data.
1. generate sample and label.
2. get the batch size.
"""
train_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_image_clip'

static_state = []
label_static_state = [] # label 0
normal_navigation = []
label_normal_navigation = [] #label 1
maneuvring_operation = []
label_maneuvring_operation = [] #label 2

# Step 1. get the train_dir file address and save all data into the list
def get_files(file_dir,ratio):
    for file in os.listdir(file_dir+'/result-static'):
        static_state.append(file_dir+'/result-static'+file)
        label_static_state.append(0)
    for file in os.listdir(file_dir+'/result-normal-navigation'):
        normal_navigation.append(file_dir+'/result-normal-navigation'+file)
        label_normal_navigation.append(1)
    for file in os.listdir(file_dir+'/result-maneuvring'):
        maneuvring_operation.append(file_dir+'/result-maneuvring'+file)
        label_maneuvring_operation.append(2)

# step 2: hstack the label and image
    image_list = np.hstack((static_state,normal_navigation,maneuvring_operation))
    label_list = np.hstack((label_static_state,label_normal_navigation,label_maneuvring_operation))

    #using shuffle
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # transfer all the img and label into list
    all_image_list = list(temp[:,0])
    all_label_list = list(temp[:,1])

    #divide the list inot two parts: one part for training, and another part for validation
    #ratio is the proportion of validation dataset
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample*ratio)) # the total number of validation
    n_train = n_sample - n_val # the total number of training

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels

# generate the batch
#step1. input the above list into get_batch(), and tranfer the types and produce a input queue
# due to the img and lab seperable, using tf.train.slice_input_producer(), and then using tf.read_file()
# to read the images
# image_W, image_H: set the height and width of image
# batch_size: for the number of images for every batches
# capacity: the maximum volumn of a queue
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    # transfer the type
    image = tf.cast(image,tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0]) # read img from a queue

# step2. decoding the image
    image = tf.image.decode_jpeg(image_contents,channels=3)

#step3. generation batch
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size=batch_size,
                                             num_threads=32,
                                             capacity=capacity)
    # rearrange label, the row num is [batch_size]
    label_batch = tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch,tf.float32)
    return image_batch,label_batch









