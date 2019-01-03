#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 12/22/2018 3:38 PM
# @Author : Xiang Chen (Richard)
# @File : network_model.py
# @Software: VS code
import tensorflow as tf
import numpy as np

#define the read and decode network
def read_and_decode_tfrecords(example):
    """
    Transfer the image data into the standard tensorflow dimensions
    """
    save_image_label_dict = {}
    save_image_label_dict['label_1'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    save_image_label_dict['label_2'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    save_image_label_dict['label_3'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    parsed = tf.parse_single_example(example, features=save_image_label_dict)
    image = tf.decode_raw(parsed['raw_image'], out_type=tf.uint8)
    image = tf.reshape(image, shape=[360, 490, 3])  #shape = (490,360,3) one image size
    image = tf.image.per_image_standardization(image)  # standarization the image and accelerate the training process
    label_1 = parsed['label_1']  # label1 for static or anchorage state
    label_2 = parsed['label_2']  # label2 for normal naivgation
    label_3 = parsed['label_3']  # label3 for maneuvring operation
    label_1 = tf.cast(label_1, tf.int32)  # change the label1 data type into int32
    label_2 = tf.cast(label_2, tf.int32)  # change the label2 data type into int32
    label_3 = tf.cast(label_3, tf.int32)  # change the label3 data type into int32
    return image, label_1, label_2, label_3

def weight_variable(shape):
    initial = tf.random_normal(shape=conv_shape, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.zeros(shape=bias_shape)
    return tf.Variable(initial)

x = tf.placeholder(dtype=tf.float32, shape=[None, 360, 490, 3])
y1_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])
y2_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])
y3_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])

# first layer




