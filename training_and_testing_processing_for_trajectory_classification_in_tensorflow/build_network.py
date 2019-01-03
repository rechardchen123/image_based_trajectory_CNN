#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 12/29/2018 8:41 PM
# @Author : Xiang Chen (Richard)
# @File : build_network.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import os

def convolution_layer(image_tensor,batch_size,n_classes):
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
        shape1 = tf.shape(conv)
        rank1 = tf.rank(conv)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        shape2 = tf.shape(conv1)
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(
            conv1,
            ksize=[1, 3, 3, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pooling1')
        shape3 = tf.shape(pool1)
        norm1 = tf.nn.lrn(
            pool1,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm1')
        shape4 = tf.shape(norm1)
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
        shape5 = tf.shape(pre_activation)
        conv2 = tf.nn.relu(pre_activation, name='conv2')
        shape6 = tf.shape(conv2)
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(
            conv2,
            depth_radius=4,
            bias=1.0,
            alpha=0.001 / 9.0,
            beta=0.75,
            name='norm2')
        shape7 = tf.shape(norm2)
        pool2 = tf.nn.max_pool(
            norm2,
            ksize=[1, 3, 3, 1],
            strides=[1, 1, 1, 1],
            padding='SAME',
            name='pooling2')
        shape8 = tf.shape(pool2)
    with tf.variable_scope('conv3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        shape9 = tf.shape(reshape)
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
        shape10 = tf.shape(local3)
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
        shape11 = tf.shape(local4)
    with tf.variable_scope('dropout') as scope:
        drop_out = tf.nn.dropout(local4, 0.8)
        shape12 = tf.shape(drop_out)
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
            tf.matmul(local4, weights), biases, name='softmax_linear')
        shape13 = tf.shape(softmax_linear)
        return softmax_linear


def evaluation(softmax_linear, label1_tensor, label2_tensor, label3_tensor,learning_rate):
    with tf.variable_scope('loss') as scope:
        y1 = tf.nn.sigmoid(softmax_linear, name='label1_softmax')
        y2 = tf.nn.sigmoid(softmax_linear, name='label2_softmax')
        y3 = tf.nn.sigmoid(softmax_linear, name='label3_softmax')
        y1_1 = tf.clip_by_value(
            y1, 1e-8, tf.reduce_max(y1), name='label1_softmax_clip')
        y2_1 = tf.clip_by_value(
            y2, 1e-8, tf.reduce_max(y2), name='label2_softmax_clip')
        y3_1 = tf.clip_by_value(
            y3, 1e-8, tf.reduce_max(y3), name='label3_softmax_clip')
        # loss1 = tf.reduce_mean(-tf.reduce_sum(label1_tensor*tf.log(y1_1)))
        # loss2 = tf.reduce_mean(-tf.reduce_sum(label2_tensor*tf.log(y2_1)))
        # loss3 = tf.reduce_mean(-tf.reduce_sum(label1_tensor*tf.log(y1_1)))
        cross_entropy1 = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label1_tensor, logits=y1_1, name='xentropy_per_label1')
        cross_entropy2 = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label2_tensor, logits=y2_1, name='xentropy_per_label2')
        cross_entropy3 = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label3_tensor, logits=y3_1, name='xentropy_per_label3')
        loss1 = tf.reduce_mean(cross_entropy1, name='loss1')
        loss2 = tf.reduce_mean(cross_entropy2, name='loss2')
        loss3 = tf.reduce_mean(cross_entropy3, name='loss3')
        loss = (loss1 + loss2 + loss3) / 3
        tf.summary.scalar('loss', loss)

        #optimization
        train_op=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.variable_scope('accuracy') as scope:
            correct_predict1 = tf.equal(tf.argmax(label1_tensor), tf.argmax(y1))
            correct_predict2 = tf.equal(tf.argmax(label2_tensor), tf.argmax(y2))
            correct_predict3 = tf.equal(tf.argmax(label3_tensor), tf.argmax(y3))
            correct_predict1 = tf.cast(correct_predict1, tf.float32)
            correct_predict2 = tf.cast(correct_predict2, tf.float32)
            correct_predict3 = tf.cast(correct_predict3, tf.float32)
            accuracy1 = tf.reduce_mean(correct_predict1)
            accuracy2 = tf.reduce_mean(correct_predict2)
            accuracy3 = tf.reduce_mean(correct_predict3)
            accuracy = (accuracy1 + accuracy2 + accuracy3) / 3
            tf.summary.scalar('accuracy', accuracy)
    return loss, accuracy


# def training(loss, learning_rate_tensor):
#     with tf.name_scope('optimizer'):
#         train_op = tf.train.AdamOptimizer(learning_rate=learning_rate_tensor).minimize(loss)
#     return train_op
#
# def evaluation(label1_tensor, label2_tensor, label3_tensor, y1, y2, y3):
#     with tf.variable_scope('accuracy') as scope:
#         correct_predict1 = tf.equal(tf.argmax(label1_tensor), tf.argmax(y1))
#         correct_predict2 = tf.equal(tf.argmax(label2_tensor), tf.argmax(y2))
#         correct_predict3 = tf.equal(tf.argmax(label3_tensor), tf.argmax(y3))
#         correct_predict1 = tf.cast(correct_predict1, tf.float32)
#         correct_predict2 = tf.cast(correct_predict2, tf.float32)
#         correct_predict3 = tf.cast(correct_predict3, tf.float32)
#         accuracy1 = tf.reduce_mean(correct_predict1)
#         accuracy2 = tf.reduce_mean(correct_predict2)
#         accuracy3 = tf.reduce_mean(correct_predict3)
#         accuracy = (accuracy1 + accuracy2 + accuracy3) / 3
#         tf.summary.scalar('accuracy', accuracy)
#     return accuracy
