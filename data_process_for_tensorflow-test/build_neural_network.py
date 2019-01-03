#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/21/2018 2:57 PM 
# @Author : Xiang Chen (Richard)
# @File : build_neural_network.py 
# @Software: PyCharm
from data_process_for_tensorflow import image_data_preprocessing
import tensorflow as tf

'''
define the network:
# input parameters: images, image_batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
# return parameters: logits, float, [batch_size, n_classes]
'''

def inference(images, batch_size, n_classes):
    # convolutional layer 1
    # 3X3 convolutional kernal(total number 64), padding = 'SAME', it means the same size
    # with the original images after padding operation. Activation function is relu()
    with tf.variable_scope('conv1') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], stddev=1.0, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[64]),
                             name='biases', dtype=tf.float32)
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    # pooling layer1
    # 3X3 maximum pooling layer, the stride is 2, after pooling operation, run lrn()
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',
                               name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # convolution layer 2
    # 3 X 3 convolutional kernals (total 16), padding = 'SAME', it means the same size
    # with the original images after padding operation. Activation function is relu()
    with tf.variable_scope('conv2') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 16], stddev=0.1, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[16]),
                             name='biases', dtype=tf.float32)
        conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pooling layer 2
    # 3X3 maximum pooling layer, the stride is 2, after pooling operation, run lrn()
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')

    # fully connect layer 3
    # 128 neural kernals
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.Variable(tf.truncated_normal(shape=[dim, 128], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                             name='biases', dtype=tf.float32)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # fully connect layer 4
    # 128 neural kernals
    with tf.variable_scope('local4') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128, 128], stddev=0.005, dtype=tf.float32),
                              name='weights', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[128]),
                             name='biases', dtype=tf.float32)
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # dropout layer
    with tf.variable_scope('dropout') as scope:
        drop_out = tf.nn.dropout(local4, 0.8)

    # softmax layer
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.Variable(tf.truncated_normal(shape=[128, n_classes], stddev=0.005, dtype=tf.float32),
                              name='softmax_linear', dtype=tf.float32)
        biases = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=[n_classes]),
                             name='biases', dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    return softmax_linear


# loss calculate
def loss(logits, labels):
    """
    Input parameters: the network output value: logits
    Input parameters: the labels
    return: loss values
    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


# loss value optimization
def training(loss, learning_rate):
    """
    loss value optimization
    Input parameter:
    loss: loss value
    learning_rate: the ration of learning
    Return:
        train_op: traing op to run it using sess.run
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    """
    Evaluation for the network
    Input parameters:
        logits: the output value of the CNN
        labels: the true value
    Return:
        accuracy: the correct ratio
    """
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        correct = tf.cast(correct, tf.float16)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
