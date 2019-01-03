#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 12/29/2018 9:36 PM
# @Author : Xiang Chen (Richard)
# @File : training_network.py
# @Software: PyCharm
import os
import glob
import numpy as np
import tensorflow as tf
import build_network

N_CLASSES = 3
MAX_STEP = 60000
BATCH_SIZE = 32
LEARNING_RATE = 0.0001


def read_and_decode(record):
    save_image_label_dict = {}
    save_image_label_dict['raw_image'] = tf.FixedLenFeature(
        shape=[], dtype=tf.string)
    save_image_label_dict['label_1'] = tf.FixedLenFeature(
        shape=[3], dtype=tf.int64)
    save_image_label_dict['label_2'] = tf.FixedLenFeature(
        shape=[3], dtype=tf.int64)
    save_image_label_dict['label_3'] = tf.FixedLenFeature(
        shape=[3], dtype=tf.int64)
    parsed = tf.parse_single_example(record, features=save_image_label_dict)
    image = tf.decode_raw(parsed['raw_image'], out_type=tf.uint8)
    image = tf.reshape(image, shape=[360, 490, 3])
    # standarization the image and accelerate the training process
    image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32)
    label_1 = parsed['label_1']  # label1 for static or anchorage state
    label_2 = parsed['label_2']  # label2 for normal naivgation
    label_3 = parsed['label_3']  # label3 for maneuvring operation
    label_1 = tf.cast(label_1,
                      tf.int32)  # change the label1 data type into int32
    label_2 = tf.cast(label_2,
                      tf.int32)  # change the label2 data type into int32
    label_3 = tf.cast(label_3,
                      tf.int32)  # change the label3 data type into int32
    return image, label_1, label_2, label_3


with tf.name_scope('input_layer'):
    image_tensor = tf.placeholder(
        dtype=tf.float32, shape=[BATCH_SIZE, 360, 490, 3], name='input_image')
    label1_tensor = tf.placeholder(
        dtype=tf.float32, shape=[BATCH_SIZE, 3], name='label1')
    label2_tensor = tf.placeholder(
        dtype=tf.float32, shape=[BATCH_SIZE, 3], name='label2')
    label3_tensor = tf.placeholder(
        dtype=tf.float32, shape=[BATCH_SIZE, 3], name='label3')

    train_logit = build_network.convolution_layer(image_tensor, BATCH_SIZE,
                                                  N_CLASSES)
    train_evaluation = build_network.evaluation(train_logit, label1_tensor,
                                                label2_tensor, label3_tensor,
                                                LEARNING_RATE)
    filenames = glob.glob(
        r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset\train.tfrecords'
    )
    logs_train_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset'
    train_dataset = tf.data.TFRecordDataset(filenames)
    train_dataset = train_dataset.map(read_and_decode)
    # batch_size_tensor  = tf.convert_to_tensor(BATCH_SIZE,tf.int64)
    train_dataset = train_dataset.batch(
        batch_size=BATCH_SIZE, drop_remainder=True)
    train_iter = train_dataset.make_one_shot_iterator()
    train_next_element = train_iter.get_next()
    init = tf.global_variables_initializer()
    writer = tf.summary.FileWriter(logs_train_dir, tf.get_default_graph())
    # summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    with tf.Session() as session:
        session.run(init)
        count = 0
        try:
            while True:
                image, label1, label2, label3 = session.run(train_next_element)
                tra_loss, tra_acc = session.run(
                    [train_logit, train_evaluation],
                    feed_dict={
                        image_tensor: image,
                        label1_tensor: label1,
                        label2_tensor: label2,
                        label3_tensor: label3
                    })
                if count % 20 == 0:
                    print('train loss=', np.around(tra_loss, 2))
                    # print('train accuracy = ', np.multiply(tra_acc,100.0))
                    print('train accuracy = ', tra_acc)
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(session, checkpoint_path, global_step=count)
                count += 1
        except tf.errors.OutOfRangeError:
            print('end!')
