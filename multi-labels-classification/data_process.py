#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/22/2018 3:12 PM 
# @Author : Xiang Chen (Richard)
# @File : data_process.py 
# @Software: PyCharm
import os
import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_PATH = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset\AIS_trajectory_image_clip_labels'
IMAGE_LABEL_PATH = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset'

train_list = []
test_list = []

# open the files
with open(IMAGE_LABEL_PATH + "\label.txt") as f:
    i = 1
    for line in f.readlines():
        if i % 30 == 0:
            test_list.append(line)
        else:
            train_list.append(line)
        i += 1

np.random.shuffle(train_list)
np.random.shuffle(test_list)


# transfer the labels
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


def image_to_tfrecords(list, tf_record_path):
    '''
    This funciton is used to tranfer the original trajectory images into tfrecord format for
    tensorflow training.
    :param list: Image list
    :param tf_record_path: save path
    '''
    tf_write = tf.python_io.TFRecordWriter(tf_record_path)
    for i in range(len(list)):
        item = list[i]
        item = item.strip('\n') # use strip() to delete the start and end string '\n'
        items = item.split(',') # slice method using ','
        image_name = items[0]
        image_path = os.path.join(IMAGE_PATH, image_name)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            image = image.tobytes()
            features = {}
            features['raw_image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
            labels = int_to_one_hot(items[1:])
            features['label_1'] = tf.train.Feature(int64_list=tf.train.Int64List(value=labels[0]))
            features['label_2'] = tf.train.Feature(int64_list=tf.train.Int64List(value=labels[1]))
            features['label_3'] = tf.train.Feature(int64_list=tf.train.Int64List(value=labels[2]))
            tf_features = tf.train.Features(feature=features)
            example = tf.train.Example(features=tf_features)  # protocol buffer
            tf_serialized = example.SerializeToString()
            tf_write.write(tf_serialized)
        else:
            print("not")
    tf_write.close()

image_to_tfrecords(train_list, r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset\train.tfrecords')
image_to_tfrecords(test_list, r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset\test.tfrecords')
