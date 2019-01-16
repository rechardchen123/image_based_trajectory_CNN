#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 12/29/2018 8:41 PM
# @Author : Xiang Chen (Richard)
# @File : build_network.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


def convolution_layer(image_tensor, batch_size, n_classes):
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(
            tf.truncated_normal(
                shape=[3, 3, 3, 64], stddev=1.0, dtype=tf.float32),
            name='weights',
            dtype=tf.float32)
        biases = tf.Variable(
            tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
            name='biases',
            dtype=tf.float32)
        conv = tf.nn.conv2d(
            image_tensor, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pooling1')
        norm1 = tf.nn.lrn(
            pool1,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm1')
    # convolutional layer 2
    # 3 X 3 convolutional kernals and total number 16
    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(
            tf.truncated_normal(
                shape=[3, 3, 64, 16], stddev=0.1, dtype=tf.float32),
            name='weights',
            dtype=tf.float32)
        biases = tf.Variable(
            tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
            name='biases',
            dtype=tf.float32)
        conv = tf.nn.conv2d(
            norm1,
            weights,
            strides=[1, 1, 1, 1],
            padding='SAME',
        )
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(
            conv2,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm2')
        pool2 = tf.nn.max_pool(
            norm2,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pooling2')
    with tf.variable_scope('conv3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(
            tf.truncated_normal(
                shape=[dim, 128], stddev=0.005, dtype=tf.float32),
            name='weights',
            dtype=tf.float32)
        biases = tf.Variable(
            tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
            name='biases',
            dtype=tf.float32)
        local3 = tf.nn.relu(
            tf.matmul(reshape, weights) + biases, name=scope.name)
    # fully connected layer
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(
            tf.truncated_normal(
                shape=[128, 128], stddev=0.005, dtype=tf.float32),
            name='weights',
            dtype=tf.float32)
        biases = tf.Variable(
            tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
            name='biases',
            dtype=tf.float32)
        local4 = tf.nn.relu(
            tf.matmul(local3, weights) + biases, name=scope.name)
    with tf.variable_scope('dropout') as scope:
        drop_out = tf.nn.dropout(local4, 0.7)
    with tf.variable_scope('softmax_layer') as scope:
        weights = tf.Variable(
            tf.truncated_normal(
                shape=[128, n_classes], stddev=0.005, dtype=tf.float32),
            name='weights',
            dtype=tf.float32)
        biases = tf.Variable(
            tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),
            name='biases',
            dtype=tf.float32)
        softmax_linear = tf.add(
            tf.matmul(drop_out, weights), biases, name='softmax_linear')
        return softmax_linear

# def convolution_layer(image_tensor, batch_size, n_classes):
#     with tf.variable_scope('conv1_1') as scope:
#         weight1_1 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 3, 64], dtype=tf.float32), name='weight1-1', dtype=tf.float32)
#         bias1_1 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[64]), name='bias1-1', dtype=tf.float32)
#         conv = tf.nn.conv2d(image_tensor, weight1_1, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias1_1)
#         conv1_1 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('conv1_2') as scope:
#         weight1_2 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 64, 64], dtype=tf.float32), name='weight1-2', dtype=tf.float32)
#         bias1_2 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[64]), name='bias1-1', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv1_1, weight1_2, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias1_2)
#         conv1_2 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('max_pooling1') as scope:
#         pool1 = tf.nn.max_pool(
#             conv1_2,
#             ksize=[1, 2, 2, 1],
#             strides=[1, 2, 2, 1],
#             padding='SAME',
#             name='max_pooling1')
#     with tf.variable_scope('conv2_1') as scope:
#         weight2_1 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 64, 128], dtype=tf.float32), name='weight2_1', dtype=tf.float32)
#         bias2_1 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[128]), name='bias2_1', dtype=tf.float32)
#         conv = tf.nn.conv2d(pool1, weight2_1, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias2_1)
#         conv2_1 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('conv2_2') as scope:
#         weight2_2 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 128, 128], dtype=tf.float32), name='weight2_2', dtype=tf.float32)
#         bias2_2 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[128]), name='bias2_1', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv2_1, weight2_2, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias2_2)
#         conv2_2 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('max_pooling2') as scope:
#         pool2 = tf.nn.max_pool(
#             conv2_2,
#             ksize=[1, 2, 2, 1],
#             strides=[1, 2, 2, 1],
#             padding='SAME',
#             name='max_pooling2')
#     with tf.variable_scope('conv3-1') as scope:
#         weight3_1 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 128, 256], dtype=tf.float32), name='weight3_1', dtype=tf.float32)
#         bias3_1 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[256]), name='bias2_1', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv2_2, weight3_1, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias3_1)
#         conv3_1 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('conv3-2') as scope:
#         weight3_2 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 256, 256], dtype=tf.float32), name='weight3_2', dtype=tf.float32)
#         bias3_2 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[256]), name='bias2_1', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv3_1, weight3_2, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias3_2)
#         conv3_2 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('conv3-3') as scope:
#         weight3_3 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 256, 256], dtype=tf.float32), name='weight3_3', dtype=tf.float32)
#         bias3_3 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[256]), name='bias3_3', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv3_2, weight3_3, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias3_3)
#         conv3_3 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('max_pooling3') as scope:
#         pool3 = tf.nn.max_pool(
#             conv3_3,
#             ksize=[1, 2, 2, 1],
#             strides=[1, 2, 2, 1],
#             padding='SAME',
#             name='max_pooling2')
#     with tf.variable_scope('conv4-1') as scope:
#         weight4_1 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 256, 512], dtype=tf.float32), name='weight4_1', dtype=tf.float32)
#         bias4_1 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[512]), name='bias4_1', dtype=tf.float32)
#         conv = tf.nn.conv2d(pool3, weight4_1, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias4_1)
#         conv4_1 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('conv4-2') as scope:
#         weight4_2 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 512, 512], dtype=tf.float32), name='weight4_2', dtype=tf.float32)
#         bias4_2 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[512]), name='bias4_2', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv4_1, weight4_2, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias4_2)
#         conv4_2 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('conv4-3') as scope:
#         weight4_3 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 512, 512], dtype=tf.float32), name='weight4_3', dtype=tf.float32)
#         bias4_3 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[512]), name='bias4_2', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv4_2, weight4_3, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias4_3)
#         conv4_3 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('max_pooling4') as scope:
#         pool4 = tf.nn.max_pool(
#             conv4_3,
#             ksize=[1, 2, 2, 1],
#             strides=[1, 2, 2, 1],
#             padding='SAME',
#             name='max_pooling4')
#     with tf.variable_scope('conv5-1') as scope:
#         weight5_1 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 512, 512], dtype=tf.float32), name='weight5_1', dtype=tf.float32)
#         bias5_1 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[512]), name='bias5_1', dtype=tf.float32)
#         conv = tf.nn.conv2d(pool4, weight5_1, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias5_1)
#         conv5_1 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('conv5-2') as scope:
#         weight5_2 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 512, 512], dtype=tf.float32), name='weight5_2', dtype=tf.float32)
#         bias5_2 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[512]), name='bias5_2', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv5_1, weight5_2, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias5_2)
#         conv5_2 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('conv5-3') as scope:
#         weight5_3 = tf.Variable(
#             tf.random_normal(
#                 shape=[3, 3, 512, 512], dtype=tf.float32), name='weight5_3', dtype=tf.float32)
#         bias5_3 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[512]), name='bias5_3', dtype=tf.float32)
#         conv = tf.nn.conv2d(conv5_2, weight5_3, strides=[1, 1, 1, 1], padding='SAME')
#         out = tf.nn.bias_add(conv, bias5_3)
#         conv5_3 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('max_pooling5') as scope:
#         pool5 = tf.nn.max_pool(
#             conv5_3,
#             ksize=[1, 2, 2, 1],
#             strides=[1, 2, 2, 1],
#             padding='SAME',
#             name='max_pooling5')
#     pool5 = tf.reshape(pool5, shape=[batch_size,-1])
#     dim = pool5.get_shape()[1].value
#     with tf.variable_scope('fully_connected1') as scope:
#         weight_fc1 = tf.Variable(
#             tf.random_normal(
#                 shape=[dim, 4096], dtype=tf.float32), name='weight_fc1', dtype=tf.float32)
#         bias_fc1 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[4096]), name='bias_fc1', dtype=tf.float32)
#         fc = tf.matmul(pool5, weight_fc1)
#         out = tf.nn.bias_add(fc, bias_fc1)
#         fc1 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('fully_connected2') as scope:
#         weight_fc2 = tf.Variable(
#             tf.random_normal(
#                 shape=[4096, 4096], dtype=tf.float32), name='weight_fc1', dtype=tf.float32)
#         bias_fc2 = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[4096]), name='bias_fc2', dtype=tf.float32)
#         fc = tf.matmul(fc1, weight_fc2)
#         out = tf.nn.bias_add(fc, bias_fc2)
#         fc2 = tf.nn.relu(out, name=scope.name)
#     with tf.variable_scope('last_fc') as name:
#         weight_fcl = tf.Variable(
#             tf.random_normal(
#                 shape=[4096, n_classes], dtype=tf.float32), name='weight_fcl', dtype=tf.float32)
#         bias_fcl = tf.Variable(
#             tf.constant(
#                 value=0.1, dtype=tf.float32, shape=[n_classes]), name='bias_fcl', dtype=tf.float32)
#         fc = tf.matmul(fc2, weight_fcl)
#         softmax_linear = tf.nn.bias_add(fc, bias_fcl)
#     return softmax_linear


def evaluation(softmax_linear, label1_tensor, label2_tensor, label3_tensor,
               learning_rate):
    with tf.name_scope('loss'):
        y1 = tf.nn.softmax(softmax_linear, name='label1_softmax')
        y2 = tf.nn.softmax(softmax_linear, name='label2_softmax')
        y3 = tf.nn.softmax(softmax_linear, name='label3_softmax')
        # y1 = tf.nn.softmax(softmax_linear, name='label1_softmax')
        # y2 = tf.nn.softmax(softmax_linear, name='label2_softmax')
        # y3 = tf.nn.softmax(softmax_linear, name='label3_softmax')
        y1_1 = tf.clip_by_value(
            y1, 1e-8, tf.reduce_max(y1), name='label1_softmax_clip')
        y2_1 = tf.clip_by_value(
            y2, 1e-8, tf.reduce_max(y2), name='label2_softmax_clip')
        y3_1 = tf.clip_by_value(
            y3, 1e-8, tf.reduce_max(y3), name='label3_softmax_clip')
        # cross_entropy1 = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=label1_tensor, logits=y1_1, name='xentropy_per_label1')
        # cross_entropy2 = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=label2_tensor, logits=y2_1, name='xentropy_per_label2')
        # cross_entropy3 = tf.nn.sigmoid_cross_entropy_with_logits(
        #     labels=label3_tensor, logits=y3_1, name='xentropy_per_label3')
        cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(
            labels=label1_tensor, logits=y1_1, name='xentropy_per_label1')
        cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(
            labels=label2_tensor, logits=y2_1, name='xentropy_per_label2')
        cross_entropy3 = tf.nn.softmax_cross_entropy_with_logits(
            labels=label3_tensor, logits=y3_1, name='xentropy_per_label3')
        loss1 = tf.reduce_mean(cross_entropy1, name='loss1')
        loss2 = tf.reduce_mean(cross_entropy2, name='loss2')
        loss3 = tf.reduce_mean(cross_entropy3, name='loss3')
        # loss1 = tf.reduce_mean(-tf.reduce_sum(label1_tensor * tf.log(y1_1)))
        # loss2 = tf.reduce_mean(-tf.reduce_sum(label2_tensor * tf.log(y2_1)))
        # loss3 = tf.reduce_mean(-tf.reduce_sum(label3_tensor * tf.log(y3_1)))
        loss = (loss1 + loss2 + loss3) / 3
        tf.summary.scalar('total loss', loss)
        tf.summary.histogram('loss', loss)
        # optimization
    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(loss)

    with tf.name_scope('accuracy'):
        correct_predict1 = tf.equal(
            tf.argmax(label1_tensor), tf.argmax(y1))
        correct_predict2 = tf.equal(
            tf.argmax(label2_tensor), tf.argmax(y2))
        correct_predict3 = tf.equal(
            tf.argmax(label3_tensor), tf.argmax(y3))
        correct_predict1 = tf.cast(correct_predict1, tf.float32)
        correct_predict2 = tf.cast(correct_predict2, tf.float32)
        correct_predict3 = tf.cast(correct_predict3, tf.float32)
        accuracy1 = tf.reduce_mean(correct_predict1, name='accuracy1')
        accuracy2 = tf.reduce_mean(correct_predict2, name='accuracy2')
        accuracy3 = tf.reduce_mean(correct_predict3, name='accuracy3')
        accuracy = (accuracy1 + accuracy2 + accuracy3) / 3
        tf.summary.scalar('total accuracy', accuracy)
        tf.summary.histogram('accuracy', accuracy)
    return loss,loss1,loss2,loss3, accuracy,accuracy1,accuracy2,accuracy3
