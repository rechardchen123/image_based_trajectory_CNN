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
    if labels[0] == -1:
        label.append([0, 0, 0])
    else:
        label.append([1, 0, 0])
    if labels[1] == -1:
        label.append([0, 0, 0])
    else:
        label.append([0, 1, 0])
    if labels[2] == -1:
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
    labels = []
    #read the labels
    with open (label_dir +r'\label.txt') as f:
        for line in f.readlines():
            labels.append(line)
    # transfer the labels using the function int_to_one_hot
    for i in range(len(labels)):
        item = labels[i]
        item = item.strip('\n')  # use strip() to delete the start and end string '\n'
        items = item.split(',')  # slice method using ','
        #save the different motion mode into the different lists
        if items[1]==1:
            # it means the static/anchorage state for the vessel
            label_static_state.append(int_to_one_hot(items[1:]))
            image_name_static = items[0]
            image_path_static = os.path.join(file_dir, image_name_static)
            for file in os.listdir(image_path_static):
                static_state.append(file)
        elif items[2] == 1:
            # it means the normal navigation state for vessel
            label_normal_navigation.append(int_to_one_hot(items[1:])
            image_name_normal_navigation = items[0]
            image_path_normal_navigation = os.path.join(file_dir,image_name_normal_navigation)
            for file in os.listdir(image_path_normal_navigation):
                normal_navigation.append(file)
        elif items[3] == 1:
            # it means the maneuvring operation 
            label_maneuvring_operation.append(int_to_one_hot(items[1:]))
            image_name_maneuvring = items[0]
            image_path_maneuvring = os.path.join(file_dir,image_name_maneuvring)
            for file in os.listdir(image_path_maneuvring):
                maneuvring_operation.append(file)
    # step 2: hstack the label and image
    image_list = np.hstack((static_state,normal_navigation,maneuvring_operation))
    label_list = np.hstack((label_static_state,label_normal_navigation,label_maneuvring_operation))

    # using shuffle
    temp = np.array([image_list,label_list])
    temp = np.transpose()
    np.random.shuffle(temp)
    # transfer all the img and label into list
    all_image_list = list(temp[:,0])
    all_label_list = list(temp[:, 1])
    

