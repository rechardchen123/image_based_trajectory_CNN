#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/22/2018 3:38 PM 
# @Author : Xiang Chen (Richard)
# @File : network_model.py 
# @Software: PyCharm
import tensorflow as tf
import numpy as np

# define the data decoder api
def parse_tf(example):
    """
    Transfer the image data into the standard tensorflow dimensions
    """
    dics = {}
    dics['label_1'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    dics['label_2'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    dics['label_3'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    parsed = tf.parse_single_example(example, features=dics)
    image = tf.decode_raw(parsed['raw_image'], out_type=tf.uint8)
    image = tf.reshape(image, shape=[490, 360, 3])
    image = tf.image.per_image_standardization(image)
    label_1 = parsed['label_1']
    label_2 = parsed['label_2']
    label_3 = parsed['label_3']

    label_1 = tf.cast(label_1, tf.int32)
    label_2 = tf.cast(label_2, tf.int32)
    label_3 = tf.cast(label_3, tf.int32)

    return image, label_1, label_2, label_3


def Conv(x, conv_shape, bias_shape, parameters, padding="SAME", strides=[1, 1, 1, 1]):
    w = tf.Variable(initial_value=tf.random_normal(shape=conv_shape, dtype=tf.float32), trainable=False)
    b = tf.Variable(initial_value=tf.zeros(shape=bias_shape), trainable=False)
    parameters += [w, b]
    conv1 = tf.nn.conv2d(x, w, strides=strides, padding=padding)
    out = tf.nn.bias_add(conv1, b)
    return tf.nn.relu(out)


def Max_Pooling(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME"):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def FC(x, w_shape, b_shape, parameters):
    w = tf.Variable(initial_value=tf.random_normal(shape=w_shape, dtype=tf.float32))
    b = tf.Variable(initial_value=tf.zeros(shape=b_shape))
    parameters += [w, b]
    fc = tf.matmul(x, w)
    fc = tf.nn.bias_add(fc, b)
    return tf.nn.relu(fc)


def Last_FC(x, w_shape, b_shape):
    w = tf.Variable(initial_value=tf.random_normal(shape=w_shape, dtype=tf.float32))
    b = tf.Variable(initial_value=tf.zeros(shape=b_shape))
    fc = tf.matmul(x, w)
    fc = tf.nn.bias_add(fc, b)
    return fc


my_parameters = []
x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
y1_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])
y2_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])
y3_ = tf.placeholder(dtype=tf.float32, shape=[None, 3])

conv1_1 = Conv(x, conv_shape=[3, 3, 3, 64], bias_shape=[64], parameters=my_parameters)
conv1_2 = Conv(conv1_1, conv_shape=[3, 3, 64, 64], bias_shape=[64], parameters=my_parameters)
pool1 = Max_Pooling(conv1_2)

conv2_1 = Conv(pool1, conv_shape=[3, 3, 64, 128], bias_shape=[128], parameters=my_parameters)
conv2_2 = Conv(conv2_1, conv_shape=[3, 3, 128, 128], bias_shape=[128], parameters=my_parameters)
pool2 = Max_Pooling(conv2_2)

conv3_1 = Conv(pool2, conv_shape=[3, 3, 128, 256], bias_shape=[256], parameters=my_parameters)
conv3_2 = Conv(conv3_1, conv_shape=[3, 3, 256, 256], bias_shape=[256], parameters=my_parameters)
conv3_3 = Conv(conv3_2, conv_shape=[3, 3, 256, 256], bias_shape=[256], parameters=my_parameters)
pool3 = Max_Pooling(conv3_3)

conv4_1 = Conv(pool3, conv_shape=[3, 3, 256, 512], bias_shape=[512], parameters=my_parameters)
conv4_2 = Conv(conv4_1, conv_shape=[3, 3, 512, 512], bias_shape=[512], parameters=my_parameters)
conv4_3 = Conv(conv4_2, conv_shape=[3, 3, 512, 512], bias_shape=[512], parameters=my_parameters)
pool4 = Max_Pooling(conv4_3)

conv5_1 = Conv(pool4, conv_shape=[3, 3, 512, 512], bias_shape=[512], parameters=my_parameters)
conv5_2 = Conv(conv5_1, conv_shape=[3, 3, 512, 512], bias_shape=[512], parameters=my_parameters)
conv5_3 = Conv(conv5_2, conv_shape=[3, 3, 512, 512], bias_shape=[512], parameters=my_parameters)
pool5 = Max_Pooling(conv5_3)

pool5 = tf.reshape(pool5, shape=[-1, 7 * 7 * 512])

fc1 = FC(pool5, w_shape=[7 * 7 * 512, 4096], b_shape=[4096], parameters=my_parameters)
fc2 = FC(fc1, w_shape=[4096, 4096], b_shape=[4096], parameters=my_parameters)

fc3 = Last_FC(fc2, w_shape=[4096, 5], b_shape=[5])

# use three classifiers
y1 = tf.nn.softmax(fc3)
y2 = tf.nn.softmax(fc3)
y3 = tf.nn.softmax(fc3)

y1_1 = tf.clip_by_value(y1, 1e-8, tf.reduce_max(y1))
y2_1 = tf.clip_by_value(y2, 1e-8, tf.reduce_max(y2))
y3_1 = tf.clip_by_value(y3, 1e-8, tf.reduce_max(y3))

# define three loss functions
loss1 = tf.reduce_mean(-tf.reduce_sum(y1_ * tf.log(y1_1)))
loss2 = tf.reduce_mean(-tf.reduce_sum(y2_ * tf.log(y2_1)))
loss3 = tf.reduce_mean(-tf.reduce_sum(y3_ * tf.log(y3_1)))

loss = (loss1 + loss2 + loss3) / 3

train = tf.train.AdamOptimizer(learning_rate=1e-6).minimize(loss)

# define the precision
correct_predict1 = tf.equal(tf.argmax(y1_, 1), tf.argmax(y1, 1))
correct_predict2 = tf.equal(tf.argmax(y2_, 1), tf.argmax(y2, 1))
correct_predict3 = tf.equal(tf.argmax(y3_, 1), tf.argmax(y3, 1))

auc1 = tf.reduce_mean(tf.cast(correct_predict1, dtype=tf.float32))
auc2 = tf.reduce_mean(tf.cast(correct_predict2, dtype=tf.float32))
auc3 = tf.reduce_mean(tf.cast(correct_predict3, dtype=tf.float32))

auc = (auc1 + auc2 + auc3) / 3

train_dataset = tf.data.TFRecordDataset(r'C:\Users\LPT-ucesxc0\AIS-Data')
train_dataset = train_dataset.map(parse_tf)
train_dataset = train_dataset.batch(16).repeat(1)
train_iter = train_dataset.make_one_shot_iterator()
train_next_element = train_iter.get_next()

test_dataset = tf.data.TFRecordDataset(r'C:\Users\LPT-ucesxc0\AIS-Data')
test_dataset = test_dataset.map(parse_tf)
test_dataset = test_dataset.batch(16).repeat(1)
test_iter = test_dataset.make_one_shot_iterator()
test_next_element = test_iter.get_next()

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    weights = np.load(r"D:\vgg16_weight\vgg16_weights.npz")
    keys = sorted(weights.keys())
    for i, k in enumerate(keys):
        if k == 'fc8_W' or k == 'fc8_b':
            continue
        else:
            session.run(my_parameters[i].assign(weights[k]))
    count = 0
    try:
        while True:
            image, label1, label2, label3 = session.run(train_next_element)
            _, train_loss = session.run(fetches=[train, loss], feed_dict={
                x: image,
                y1_: label1,
                y2_: label2,
                y3_: label3
            })
            print("loss = ", train_loss)
            if count % 10 == 0:
                image, label1, label2, label3 = session.run(test_next_element)
                test_auc = session.run(fetches=auc, feed_dict={
                    x: image,
                    y1_: label1,
                    y2_: label2,
                    y3_: label3
                })
                print("accuracy=", test_auc)
            count += 1
    except tf.errors.OutOfRangeError:
        print("end!")
