#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/17/2018 7:39 PM 
# @Author : Xiang Chen (Richard)
# @File : clip_image.py 
# @Software: PyCharm
import os
import cv2
import glob

def CropImage4File(filepath,destpath):
    pathDir = os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join(filepath,allDir)
        dest = os.path.join(destpath,allDir)
        if os.path.isfile(child):
            image = cv2.imread(child)
            sp = image.shape # get the shape of the image and return the row and column values
            sz1 = sp[0]      # get the height of the image
            sz2 = sp[1]      # get the width of the image
            # define the clipping area
            a = 60 # x start
            b = 420 # x end
            c = 80 # y start
            d = 570 # y end

            cropImg = image[a:b,c:d] # clip the image
            cv2.imwrite(dest,cropImg) # write the image into the address


if __name__=='__main__':
    filepath = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_image'
    destpath = r'C:\Users\LPT-ucesxc0\AIS-Data\AIS_trajectory_image_clip'
    CropImage4File(filepath,destpath)

