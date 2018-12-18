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


def save_data_into_file(MMSI_list, Longitude_list, Latitude_list, Speed_list,
                        Heading_list, Day_list, time_to_seconds_list, delta_time,
                        delta_speed, delta_heading):
    """This function is for storing the data and outputing the data into a file."""
    # dictionary for storing the list and transfer it to DataFrame
    save_dict = {'MMSI': MMSI_list,
                 'Longitude': Longitude_list,
                 'Latitude': Latitude_list,
                 'Speed': Speed_list,
                 'Heading': Heading_list,
                 'Day': Day_list,
                 'time_to_seconds': time_to_seconds_list,
                 'delta_time': delta_time,
                 'delta_speed': delta_speed,
                 'delta_heading': delta_heading}
    data = pd.DataFrame(save_dict)
    # output the file
    name_mmsi = int(data.iloc[0]['MMSI'])
    name_day = int(data.iloc[0]['Day'])
    data.to_csv(str(name_mmsi) + '-' + str(name_day) + '.csv', index=False)


# read the files
file_address = glob.glob(r'C:\Users\LPT-ucesxc0\AIS-Data\test_data\205517000-6.csv')
threshold_heading_max_value = 20
for file in file_address:
    file_load = pd.read_csv(file)
    # transfer to list
    MMSI_list = list(file_load['MMSI'])
    Longitude_list = list(file_load['Longitude'])
    Latitude_list = list(file_load['Latitude'])
    Speed_list = list(file_load['Speed'])
    Heading_list = list(file_load['Heading'])
    Day_list = list(file_load['Day'])
    time_to_seconds_list = list(file_load['time_to_seconds'])
    delta_time = list(file_load['delta_time'])
    delta_speed = list(file_load['delta_speed'])
    delta_heading = list(file_load['delta_heading'])
    #loop
    index_split = file_load[file_load.delta_heading>=threshold_heading_max_value].index.tolist()
    length










