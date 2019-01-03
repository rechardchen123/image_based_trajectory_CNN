#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 12/22/2018 3:38 PM
# @Author : Xiang Chen (Richard)
# @File : network_model.py
# @Software: VS code
import tensorflow as tf
import numpy as np
import os
import glob

tf.logging.set_verbosity(tf.logging.INFO)


# define the image decoder for decoding the images and labels
def read_and_decode(filenames, batch_size, capacity, min_queue_example, num_epochs=None):
    """
    Transfer the image data into the standard tensorflow dimensions
    """
    save_image_label_dict = {}
    save_image_label_dict['raw_image'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
    save_image_label_dict['label_1'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    save_image_label_dict['label_2'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    save_image_label_dict['label_3'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    parsed = tf.parse_single_example(serialized_example, features=save_image_label_dict)
    image = tf.decode_raw(parsed['raw_image'], out_type=tf.int32)
    image = tf.reshape(image, shape=[360, 490, 3])
    # standarization the image and accelerate the training process
    image = tf.image.per_image_standardization(image)
    label_1 = parsed['label_1']  # label1 for static or anchorage state
    label_2 = parsed['label_2']  # label2 for normal naivgation
    label_3 = parsed['label_3']  # label3 for maneuvring operation
    label_1 = tf.cast(label_1, tf.int32)  # change the label1 data type into int32
    label_2 = tf.cast(label_2, tf.int32)  # change the label2 data type into int32
    label_3 = tf.cast(label_3, tf.int32)  # change the label3 data type into int32
    image_bath, label1_batch, label2_batch, label3_batch = tf.train.shuffle_batch(
        [image, label_1, label_2, label_3],
        batch_size=batch_size,
        num_threads=16,
        capacity=capacity,
        min_after_dequeue=min_queue_example)
    return image_bath, label1_batch, label2_batch, label3_batch


def Conv(x, conv_shape, bias_shape, name, padding="SAME", strides=[1, 1, 1, 1]):
    """
    This is the convolutional layer definiation.
    Input parameters:
        x: Input tensor
        conv_shape: the convolutional layer shape
        bias_shape: bias shape 
        parameters: store the weight and bias values
        padding   : padding model "SAME"
        strides   : the window for the convolution operation for every step
    Return:
        using the relu function get the value
    """
    with tf.name_scope(name):
        # tf.Variable(initializer， name), the initializer has tf.random_normal, tf.constant parameters
        # the trainable means if ture, the default, also adds the variable to the graph collection
        with tf.name_scope('weights'):
            w = tf.Variable(tf.truncated_normal(shape=conv_shape, stddev=0.1, dtype=tf.float32), dtype=tf.float32)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.constant(value=0.1, dtype=tf.float32, shape=bias_shape), dtype=tf.float32)
        # parameters = [w, b]  # combine the weights and bias
        with tf.name_scope('linear_compute'):
            conv = tf.nn.conv2d(x, w, strides=strides, padding=padding)
            out = tf.nn.bias_add(conv, b)  # using the add method to forward propagation
            tf.summary.histogram('linear', out)
        return tf.nn.relu(out, name=name)


def Max_Pooling(x, name, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"):
    """
    This is the pooling operation.
    In this, I use the maximum pooling operation.
    """
    with tf.variable_scope(name) as scope:
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding="SAME",
                              name=scope.name)


def FC(x, w_shape, b_shape, name):
    """
    This is fully connected layer definiation. 
    It maps the hidden layer characters into sample spaces.
    """
    with tf.variable_scope(name) as scope:
        w = tf.Variable(
            initial_value=tf.random_normal(shape=w_shape, dtype=tf.float32))
        b = tf.Variable(initial_value=tf.zeros(shape=b_shape))
        parameters = [w, b]
        fc = tf.matmul(x, w)
        fc = tf.nn.bias_add(fc, b)
        return tf.nn.relu(fc, name=scope.name)


def Last_FC(x, w_shape, b_shape, name):
    """
    The last fully connected layer.
    """
    with tf.variable_scope(name) as scope:
        w = tf.Variable(
            initial_value=tf.random_normal(shape=w_shape, dtype=tf.float32))
        b = tf.Variable(initial_value=tf.zeros(shape=b_shape))
        fc = tf.matmul(x, w)
        fc = tf.nn.bias_add(fc, b, name=scope.name)
        return fc


# read the data and build the palceholders
with tf.name_scope('input_layer'):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 360, 490, 3])
    tf.summary.image('input_layer_x', x)
    y1_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    tf.summary.tensor_summary('label1', y1_)
    y2_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    tf.summary.tensor_summary('label2', y2_)
    y3_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])
    tf.summary.tensor_summary('label3', y3_)

# first layer
conv1_1 = Conv(x, conv_shape=[3, 3, 3, 64], bias_shape=[64], name='Conv1_1')
conv1_2 = Conv(conv1_1, conv_shape=[3, 3, 64, 64], bias_shape=[64], name='Conv1_2')
pool1 = Max_Pooling(conv1_2, name='pool1')

# second layer
conv2_1 = Conv(pool1, conv_shape=[3, 3, 64, 128], bias_shape=[128], name='Conv2_1')
conv2_2 = Conv(conv2_1, conv_shape=[3, 3, 128, 128], bias_shape=[128], name='Conv2_2')
pool2 = Max_Pooling(conv2_2, name='pool2')

# third layer
conv3_1 = Conv(pool2, conv_shape=[3, 3, 128, 256], bias_shape=[256], name='Conv3_1')
conv3_2 = Conv(conv3_1, conv_shape=[3, 3, 256, 256], bias_shape=[256], name='Conv3_2')
conv3_3 = Conv(conv3_2, conv_shape=[3, 3, 256, 256], bias_shape=[256], name='Conv3_3')
pool3 = Max_Pooling(conv3_3, name='pool3')

# fourth layer
conv4_1 = Conv(pool3, conv_shape=[3, 3, 256, 512], bias_shape=[512], name='conv4_1')
conv4_2 = Conv(conv4_1, conv_shape=[3, 3, 512, 512], bias_shape=[512], name='conv4_2')
conv4_3 = Conv(conv4_2, conv_shape=[3, 3, 512, 512], bias_shape=[512], name='conv4_3')
pool4 = Max_Pooling(conv4_3, name='pool4')

# fifth layer
conv5_1 = Conv(pool4, conv_shape=[3, 3, 512, 512], bias_shape=[512], name='conv5_1')
conv5_2 = Conv(conv5_1, conv_shape=[3, 3, 512, 512], bias_shape=[512], name='conv5_2')
conv5_3 = Conv(conv5_2, conv_shape=[3, 3, 512, 512], bias_shape=[512], name='conv5_3')
pool5 = Max_Pooling(conv5_3, name='pool5')
pool5 = tf.reshape(pool5, shape=[-1, 4 * 4 * 512])  # the problem of the layer

# fully connected
fc1 = FC(pool5, w_shape=[4 * 4 * 512, 4096], b_shape=[4096], name='fully_connected_1')
fc2 = FC(fc1, w_shape=[4096, 4096], b_shape=[4096], name='fully_connected_2')
fc3 = Last_FC(fc2, w_shape=[4096, 3], b_shape=[3], name='last_fc')

# use three classifiers and the function is softmax
with tf.variable_scope('soft_max_layer') as scope:
    y1 = tf.nn.softmax(fc3, name='label1_softmax')
    y2 = tf.nn.softmax(fc3, name='label2_softmax')
    y3 = tf.nn.softmax(fc3, name='label3_softmax')
    y1_1 = tf.clip_by_value(y1, 1e-8, tf.reduce_max(y1), name='label1_softmax_clip')
    y2_1 = tf.clip_by_value(y2, 1e-8, tf.reduce_max(y2), name='label2_softmax_clip')
    y3_1 = tf.clip_by_value(y3, 1e-8, tf.reduce_max(y3), name='label3_softmax_clip')

# define three loss functions
with tf.variable_scope('loss') as loss:
    loss1 = tf.reduce_mean(-tf.reduce_sum(y1_ * tf.log(y1_1)), name='loss1')
    loss2 = tf.reduce_mean(-tf.reduce_sum(y2_ * tf.log(y2_1)), name='loss2')
    loss3 = tf.reduce_mean(-tf.reduce_sum(y3_ * tf.log(y3_1)), name='loss3')
    # get the mean value
    loss = (loss1 + loss2 + loss3) / 3
    tf.summary.scalar('loss', loss)

# use the AdamOptimizer to descend gradient optimization
with tf.name_scope('optimizer'):
    train = tf.train.AdamOptimizer(learning_rate=0.0005,
                                   beta1=0.9,
                                   beta2=0.999,
                                   epsilon=1e-08).minimize(loss)

# define the precision
with tf.variable_scope('accuracy') as scope:
    correct_predict1 = tf.equal(tf.argmax(y1_), tf.argmax(y1))
    correct_predict2 = tf.equal(tf.argmax(y2_), tf.argmax(y2))
    correct_predict3 = tf.equal(tf.argmax(y3_), tf.argmax(y3))
    auc1 = tf.reduce_mean(tf.cast(correct_predict1, dtype=tf.float32), name='accuracy1')
    auc2 = tf.reduce_mean(tf.cast(correct_predict2, dtype=tf.float32), name='accuracy2')
    auc3 = tf.reduce_mean(tf.cast(correct_predict3, dtype=tf.float32), name='accuracy3')
    auc = (auc1 + auc2 + auc3) / 3
    tf.summary.scalar('accuracy', auc)

# read the training tfrecords file
# train_dataset = tf.data.TFRecordDataset(r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset\train.tfrecords')
# train_dataset = tf.data.TFRecordDataset('/home/ucesxc0/Scratch/output/training_image_classification/train.tfrecords')
filenames = glob.glob(r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset\train.tfrecords')
image_batch, label1_batch, label2_batch, label3_batch = read_and_decode(filenames, 16, 200, 1)

# # get the test tfrecords files
# test_dataset = tf.data.TFRecordDataset(r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset\test.tfrecords')
# #test_dataset = tf.data.TFRecordDataset('/home/ucesxc0/Scratch/output/training_image_classification/test.tfrecords')
# test_dataset = test_dataset.map(parse_tf)
# test_dataset = test_dataset.batch(1).repeat(1)
# test_iter = test_dataset.make_one_shot_iterator()
# test_next_element = test_iter.get_next()

logs_train_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_training_dataset'
# logs_train_dir = '/home/ucesxc0/Scratch/output/training_image_classification'

# global initialization
init = tf.global_variables_initializer()
# calculate the graph
# session_config = tf.ConfigProto(
#     log_device_placement=True,
#     inter_op_parallelism_threads=0,
#     intra_op_parallelism_threads=0,
#     allow_soft_placement=True)
with tf.Session() as session:
    session.run(init)
    summary_op = tf.summary.merge_all()
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter(logs_train_dir, session.graph)
    try:
        for count in range(6000):
            image_batch, label1_batch, label2_batch, label3_batch = \
                session.run([image_batch,
                            label1_batch,
                            label2_batch,
                            label3_batch])
            summary_str, _, train_loss, train_acc = session.run(
                fetches=[summary_op, train, loss, auc],
                feed_dict={x: image_batch, y1_: label1_batch,
                           y2_: label2_batch, y3_: label3_batch})
            if count % 10 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (count, train_loss, train_acc * 100.0))
                train_writer.add_summary(summary_str, count)
            # every 100 steps, save the model
            if (count + 1) == 6000:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(session, checkpoint_path, global_step=count)
    except tf.errors.OutOfRangeError:
        print("Done training -- epoch limit reached")

