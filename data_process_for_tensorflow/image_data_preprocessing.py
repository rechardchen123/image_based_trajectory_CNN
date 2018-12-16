#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/16/2018 6:48 PM 
# @Author : Xiang Chen (Richard)
# @File : image_data_preprocessing.py 
# @Software: PyCharm

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#read the raw image, if meet 'decode uft-8' error, just use rb
#read the binary flow
#tf.gfile.FastGFile is the operation function
image_raw_data = tf.gfile.FastGFile(r'C:\Users\LPT-ucesxc0\AIS-Data\test_image\200583782-13.jpg',
                                    'rb',).read()
with tf.Session() as sess:
    #decode the image as uint8
    img_data = tf.image.decode_jpeg(image_raw_data)
    #output the image array, every number is between 0 and 255
    print(img_data.eval())
    #transfer the image into float32 and the array numbers become 0-1.
    img_data = tf.image.convert_image_dtype(img_data,dtype=tf.float32)
    print(img_data.eval())

    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    print(resized.eval())
    resized = tf.image.convert_image_dtype(resized, dtype=tf.uint8)
    encoded_image = tf.image.encode_jpeg(resized)
    #build the files and write it
    with tf.gfile.GFile('./file','wb') as f:
        f.write(encoded_image.eval())



