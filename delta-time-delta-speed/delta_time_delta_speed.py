#!/usr/bin/env python3
# _*_coding:utf-8 _*_
# @Time    :Created on Dec 04 4:39 PM 2018
# @Author  :xiang chen

import os,sys
import numpy as np
import pandas as pd
import glob
import math

def compute_time_difference(time_to_seconds_list):
    '''calculate the delta time
    Input: time_to_seconds_list.
    Output: new list for store delta time.'''
    save_time_difference = []
    for i in range(0, len(time_to_seconds_list) - 1):
        save_time_difference.append(time_to_seconds_list[i + 1] - time_to_seconds_list[i])
    save_time_difference.insert(0, 0)
    return save_time_difference

def compute_speed_difference(Speed_list):
    '''Calculate the delta speed.
    Input: Speed_list
    Output: new list for store delta speed.'''
    save_speed_difference = []
    for i in range(0, len(Speed_list) - 1):
        difference = math.fabs(Speed_list[i + 1] - Speed_list[i])
        save_speed_difference.append(difference)
    save_speed_difference.insert(0, 0.0)
    save_speed_difference1 = [round(j, 2) for j in save_speed_difference]
    return save_speed_difference1

def save_data_into_file(MMSI_list,
                        Longitude_list,
                        Latitude_list,
                        Speed_list,
                        Day_list,
                        time_to_seconds_list,
                        delta_time,
                        delta_speed):
    '''This function is for storing the data and outputing the data into a file.'''
    # dictionary for storing the list and transfer it to dataframe
    save_dict = {'MMSI':MMSI_list,
                 'Longitude':Longitude_list,
                 'Latitude':Latitude_list,
                 'Speed':Speed_list,
                 'Day':Day_list,
                 'time_to_seconds':time_to_seconds_list,
                 'delta_time':delta_time,
                 'delta_speed':delta_speed}
    data = pd.DataFrame(save_dict)
    # output the file
    name_mmsi = int(data.iloc[0]['MMSI'])
    name_day = int(data.iloc[0]['Day'])
    data.to_csv('/home/ucesxc0/Scratch/output/ais_trajectory_delta_time_delta_speed/result/%d-%d.csv' % (name_mmsi, name_day),
                index=False)


file_address = glob.glob('/home/ucesxc0/Scratch/output/ais_trajectory_delta_time_delta_speed/AIS-data-after-day-split/*.csv')
for file in file_address:
    file_load = pd.read_csv(file)
    MMSI_list = list(file_load['MMSI'])
    Longitude_list = list(file_load['Longitude'])
    Latitude_list = list(file_load['Latitude'])
    Speed_list = list(file_load['Speed'])
    Day_list = list(file_load['Day'])
    time_to_seconds_list = list(file_load['time_to_seconds'])
    # calculate the delta time and delta speed
    delta_time = compute_time_difference(time_to_seconds_list)
    delta_speed = compute_speed_difference(Speed_list)
    save_data_into_file(MMSI_list,Longitude_list,Latitude_list,Speed_list,Day_list,
                        time_to_seconds_list,delta_time,delta_speed)

