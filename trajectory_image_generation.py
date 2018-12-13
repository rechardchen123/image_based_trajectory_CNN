#!/usr/bin/env python3
# _*_coding:utf-8 _*_
# @Time    :Created on Dec 04 4:39 PM 2018
# @Author  :xiang chen
"""In this function, we three steps for generation trajectories that they contain the
motion characters.
First, generate the trajectory pictures.
Second, determine the target areas.
Third, determine the number and value of pixels of the image.
And then, output the images and generate the arraies for classifying."""

#set the environments and import packages
import os,sys
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg') #In local debugging, it should comment it and uploading to remote server,
                      # should use this.
import pandas as pd
import glob
import math
# for local debugging only.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# read the files.
trajectory_file_address = glob.glob(r"C:\Users\LPT-ucesxc0\AIS-Data\after-processed-data\*.csv")

