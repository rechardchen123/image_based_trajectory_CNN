#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/21/2018 5:25 PM 
# @Author : Xiang Chen (Richard)
# @File : test_training_model.py 
# @Software: PyCharm

from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import build_neural_network
from image_data_preprocessing import get_files

# get one image
def get_one_image(train):
    """
    Input parameters: train
    Return parameters: image
    """
    n = len(train)
    ind = np.random.randint(0,n)
    img_dir = train[ind] # random choosing one test image

    img = Image.open(img_dir)
    plt.imshow(img)
    image = np.array(img)
    return image

def evaluate_one_image(image_array):
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 3

        image = tf.cast(image_array,tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image,[1,360,490,3])

        logit = build_neural_network.inference(image,BATCH_SIZE,N_CLASSES)

        logit = tf.nn.softmax(logit)

        x= tf.placeholder(tf.float32,shape=[360,490,3])

        logs_train_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_classified_by_type'

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print('Reading checkpoints...')
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')

            prediction = sess.run(logit,feed_dict = {x:image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a static state with possibility %.6f' %prediction[:, 0])
            elif max_index==1:
                print('This is a normal navigation with possibility %.6f' %prediction[:, 1])
            else:
                print('This is a maneuvring with possibility %.6f' %prediction[:, 2])


if __name__ == '__main__':
    train_dir = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_classified_by_type'
    train,train_label,val,val_label = get_files(train_dir,0.3)
    img = get_one_image(val)
    evaluate_one_image(img)