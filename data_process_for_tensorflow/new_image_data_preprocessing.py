#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 12/16/2018 6:48 PM
# @Author : Xiang Chen (Richard)
# @File : image_data_preprocessing.py
# @Software: VS code

import tensorflow as tf
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def int_to_one_hot(labels):
    '''
    In this function, tranfer the label into one-hot coding format
    If the first element of the list is -1, it means that the vessel is not in the staic/anchorage
    state. Else, it means the vessel is now in the anchorage/static state.
    The second element of the input list is -1, it means that the vessel is not in the normal
    navigation state. Else, it means it is in the normal navigation (curise) state.
    The third element of the input list is -1, it means that the vessel is not in the maneuvring state.
    Else, it means it is in the maneuvre state.
    Input parameters: labels
    Return parameter: label, it trainfers to multi-class labels.
    '''
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


#step 1. get the images and labels
def get_files(file_dir,label_dir,ratio):
    '''
    This function is used to read the images and labels.
    Input parameters:
        file_dir: the path for storing the images
        label_dir: the path saving labels
    Return parameter:
        tra_images:training images
        tra_labels:training labels
        val_images:validation images
        val_labels:validation labels
    '''
    #define the list for saving the data
    labels = []
    static_state = []
    label_static_state =[]
    normal_navigation = []
    label_normal_navigation = []
    maneuvring_operation = []
    label_maneuvring_operation = []
    #read the labels
    with open (label_dir +r'\label.txt') as f:
        for line in f.readlines():
            labels.append(line)
    # transfer the labels using the function int_to_one_hot
    for i in range(len(labels)):
        item = labels[i].replace('\n', '')
        item = labels[i].split(',')
        if item[1] == '1' and item[2] == '-1' and item[3] == '-1':
            image_name_static = item[0]
            label_static_state.append(int_to_one_hot(item))
            image_name_static = item[0]
            image_path_static = os.path.join(file_dir, image_name_static)
            if os.path.isfile(image_path_static):
                static_image = Image.open(image_path_static)
                static_state.append(static_image)
        elif item[2] == '1' and item[3] == '-1':
            label_normal_navigation.append(int_to_one_hot(item))
            image_name_normal_navigation = item[0]
            image_path_normal_navigation = os.path.join(file_dir, image_name_normal_navigation)
            if os.path.isfile(image_path_normal_navigation):
                normal_navigation_image = Image.open(image_path_normal_navigation)
                normal_navigation.append(normal_navigation_image)
        elif item[3] == '1':
            label_maneuvring_operation.append(int_to_one_hot(item))
            image_name_maneuvring = item[0]
            image_path_maneuvring = os.path.join(file_dir, image_name_maneuvring)
            if os.path.isfile(image_path_maneuvring):
                maneuvring_image = Image.open(image_path_maneuvring)
                maneuvring_operation.append(maneuvring_image)

    # step 2: hstack the label and image
    image_list = np.hstack((static_state,normal_navigation,maneuvring_operation))
    label_list = np.hstack((label_static_state,label_normal_navigation,label_maneuvring_operation))

    # using shuffle
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # transfer all the img and label into list
    all_image_list = list(temp[:,0])
    all_label_list = list(temp[:, 1])
    # divide the list inot two parts: one part for training, and another part for validation
    # ratio is the proportion of validation dataset
    n_sample = len(all_label_list)
    n_test = int(math.ceil(n_sample*ratio)) # the total number for testing
    n_train = n_sample - n_test # the total number for training

    train_images = all_image_list[0:n_train]
    train_labels = all_label_list[0:n_train]
    train_labels = [int(float(i)) for i in train_labels]
    test_images = all_image_list[n_train:-1]
    test_labels = all_label_list[n_train:-1]
    test_labels = [int(float(i)) for i in test_labels]
    return train_images, train_labels, test_images, test_labels

def get_batch(image,label,batch_size,capacity):
    """ generate the batch
    step1. input the above list into get_batch(), and tranfer the types and produce a input queue
    due to the img and lab seperable, using tf.train.slice_input_producer(), and then using tf.read_file()
    to read the images.
    batch_size: for the number of images for every batches
    capacity: the maximum volumn of a queue"""
    # transfer the type
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2. decoding the image
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3. generation batch
    image_batch, label_batch = tf.train.batch([image, label],batch_size=batch_size,num_threads=32,
                                              capacity=capacity)
    # rearrange label, the row num is [batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch


    
            