#!/usr/bin/env python3
# _*_coding:utf-8 _*_
# @Time    :Created on Dec 04 4:39 PM 2018
# @Author  :xiang chen
import os, sys
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import pandas as pd
import glob
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def compute_time_difference(time_to_seconds_list):
    save_time_difference = []
    for i in range(0, len(time_to_seconds_list) - 1):
        save_time_difference.append(time_to_seconds_list[i + 1] - time_to_seconds_list[i])
    save_time_difference.insert(0, 0)
    return save_time_difference


def compute_speed_difference(Speed_list):
    save_speed_difference = []
    for i in range(0, len(Speed_list) - 1):
        difference = math.fabs(Speed_list[i + 1] - Speed_list[i])
        save_speed_difference.append(difference)
    save_speed_difference.insert(0, 0.0)
    save_speed_difference1 = [round(j, 2) for j in save_speed_difference]
    return save_speed_difference1


def data_compensation_algorithm_static(MMSI_list,
                                       Longitude_list,
                                       Latitude_list,
                                       Speed_list,
                                       Day_list,
                                       time_to_seconds_list,
                                       number_of_compensation,
                                       current_position):
    '''The function is about to add new static points into the trajectory sequences.
    All the list should be added.
    Input parameters: the original list
    Output: the new list contained the added points.'''
    # define the list for store the new list
    receive_MMSI = []
    receive_Longitude = []
    receive_Latitude = []
    receive_Speed = []
    receive_Day = []
    receive_time_to_seconds = []


    return receive_MMSI, receive_Longitude,\
           receive_Latitude,receive_Speed,\
           receive_Day,receive_time_to_seconds


def data_compensation_algorithm_movement(ais_data, time_threshold, maximum_time_range):
    '''This function is to add some missing data.
    The proceesing looks like this:
    1.Whether delta time > threshold and delta < one hour,
    the missing data should be compensated.
    2.else, contiue the next points.
    3.the second determines the motion patterns:
    4.if the delta V (speed increment)==0, it means the vessel is now in
    static state. The numver of compensated points is n = (Tb-Ta)/Tm.
    Tm is the sample time interval. And then add n points into the lists.
    5. else the abs(delta V)>0, the vessel is in motion status.And then, add
    some ponits into the list.
    Parameters:
        ais_data: data from the files;
        time_threshold:time interval;
        maximum_time_range:time window.
    Return:
        a new dataframe.'''


def save_data_into_file():


def clip_same_day_data():
    return null


# load the data and calculate parameters(compensate_points, list)
'''In this part, it is about to manuplate the dataset.
Firstly, transfer the dataframe into different list (MMSI list, latitude list,
Longitude list, speed list, time list etc.).
Secondly, the loop for reading the file and calculate some parameters like
compensate_points for calculating.'''

file_address = glob.glob(r'C:\Users\LPT-ucesxc0\AIS-Data\test_data\*.csv')
threshold_time = 120  # the threshold time called 120 seconds (2 minutes)
limit_time_window = 3600  # the maximum time window 1 hour
for file in file_address:
    file_load = pd.read_csv(file)
    # transfer the dataframe to list
    MMSI_list = list(file_load['MMSI'])
    Longitude_list = list(file_load['Longitude'])
    Latitude_list = list(file_load['Latitude'])
    Speed_list = list(file_load['Speed'])
    Day_list = list(file_load['Day'])
    time_to_seconds_list = list(file_load['time_to_seconds'])
    # calculate the delta time and delta speed
    delta_time = compute_time_difference(time_to_seconds_list)
    delta_speed = compute_speed_difference(Speed_list)
    # calculate the number of compensation points
    for i in range(1, len(delta_time) - 1):
        if delta_time[i] > threshold_time and delta_time[i] < limit_time_window:
            for j in range(1, len(delta_speed) - 1):
                if delta_speed[j] == 0:
                    number_of_compensation = round((time_to_seconds_list[j] -
                                                    time_to_seconds_list[j - 1]) / threshold_time)
                    current_position = j
                    data_compensation_algorithm_static(MMSI_list,
                                                       Longitude_list,
                                                       Latitude_list,
                                                       Speed_list,
                                                       Day_list,
                                                       time_to_seconds_list,
                                                       number_of_compensation,
                                                       current_position)
        elif delta_time[i] > limit_time_window:
        # clip the same day MMSI algorithm

        else:
            continue
