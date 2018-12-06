#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :Created on Dec 04 2:09 PM 2018
#@Author  :xiang chen
import os, sys
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import pandas as pd
import glob

#load the dataset

file_names = glob.glob(r'D:\Data store file\groupby-ais-file-speed-heading\*.csv')
for f in file_names:
    read_file = pd.read_csv(f)
    read_file['Day'] = pd.to_datetime(read_file['Record_Datetime']).dt.day
    groups = read_file.groupby(read_file['Day'])
    print(groups)





