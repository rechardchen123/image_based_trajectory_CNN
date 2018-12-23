#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 12/21/2018 3:58 PM
# @Author : Xiang Chen (Richard)
# @File : training_model.py
# @Software: Pycharm
import os
import numpy as np
import tensorflow as tf
from data_process_for_tensorflow import image_data_preprocessing
from data_process_for_tensorflow import build_neural_network

#define the variables
N_CLASSES = 3 # static,normal_navigation,maneuvring
BATCH_SIZE = 20
CAPACITY = 200
MAX_STEP = 200
learning_rate = 0.0001

#get the batch
train_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_classified_by_type_after_clipping'
logs_train_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_classified_by_type_after_clipping'

train, train_label,val,val_label = image_data_preprocessing.get_files(train_dir,0.3)

#training data and training labels
train_batch,train_label_batch = image_data_preprocessing.get_batch(train,train_label,BATCH_SIZE,CAPACITY)

#testing data and testing labels
val_batch, val_label_batch = image_data_preprocessing.get_batch(val,val_label,BATCH_SIZE,CAPACITY)

#training definiation
train_logits = build_neural_network.inference(train_batch,BATCH_SIZE,N_CLASSES)
train_loss = build_neural_network.loss(train_logits,train_label_batch)
train_op = build_neural_network.training(train_loss,learning_rate)
train_acc = build_neural_network.evaluation(train_logits,train_label_batch)

#testing definiation
test_logits = build_neural_network.inference(val_batch,BATCH_SIZE,N_CLASSES)
test_loss = build_neural_network.loss(test_logits,val_label_batch)
test_acc = build_neural_network.evaluation(test_logits,val_label_batch)

#the summary of log
summary_op = tf.summary.merge_all()

#produce a session
sess = tf.Session()

#produce a writer to write log files
train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)

#produce a saver to save the model
saver = tf.train.Saver()

#initialize all nodes
sess.run(tf.global_variables_initializer())

#queue surveilliance
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

#traing with batch
try:
    #get the MAX_STEP trainging,one step one batch
    for step in np.range(MAX_STEP):
        if coord.should_stop():
            break
        _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])

        #every 50 steps, print the current loss and acc, and then record log and save it to sriter
        if step % 10 ==0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%'%(step,tra_loss,tra_acc*100.0))
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str,step)
        # every 100 steps, save the model
        if (step +1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
            saver.save(sess,checkpoint_path,global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')

finally:
    coord.request_stop()

