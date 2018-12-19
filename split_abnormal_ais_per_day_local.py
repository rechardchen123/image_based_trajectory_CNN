#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/18/2018 4:12 PM 
# @Author : Xiang Chen (Richard)
# @File : split_abnormal_ais_per_day.py 
# @Software: PyCharm
import glob
import pandas as pd
import numpy as np
import math


def save_data_into_file(new_file, current_position):
    # output the file
    name_mmsi = int(new_file.iloc[0]['MMSI'])
    name_day = int(new_file.iloc[0]['Day'])
    new_file.to_csv(str(name_mmsi) + '-' + str(name_day) + '-' + str(current_position) + '.csv', index=False)


# read the files
file_address = glob.glob(r'C:\Users\LPT-ucesxc0\AIS-Data\test_data\205517000-6.csv')
threshold_heading_max_value = 20
for file in file_address:
    file_load = pd.read_csv(file)
    delta_heading = list(file_load['delta_heading'])
    # loop
    index_split = file_load[file_load.delta_heading >= threshold_heading_max_value].index.tolist()
    if len(index_split) >= 1:
        index_split.insert(0, 0)
        index_split.append(len(delta_heading))
        for i in range(0, len(index_split) - 1):
            new_file = file_load.iloc[index_split[i]:index_split[i + 1]]
            current_position = i
            save_data_into_file(new_file, current_position)
    else:
        continue
