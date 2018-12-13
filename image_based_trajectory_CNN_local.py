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

#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)


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


def linerar_interpolation_algorithm(lat0, lng0, lat1, lng1, lat):
    '''Algorithm for liner interpolation
    Input: The input number
    Output: The interpolation value.'''
    delta = (lng1 - lng0) / (lat1 - lat0)
    lng = lng0 + delta * (lat - lat0)
    return lng


def data_compensation_algorithm_static(MMSI_list,
                                       Longitude_list,
                                       Latitude_list,
                                       Day_list,
                                       delta_time,
                                       number_of_compensation,
                                       current_position):
    """The function is about to add new static points into the trajectory sequences.
    All the list should be added.
    Input parameters: the original list
    Output: the new list contained the added points."""
    incremental_time = 30 # for static traget, the time should be reported to station between 30s.
    for j in range(current_position, current_position + number_of_compensation):
        MMSI_list.insert(j, MMSI_list[0])  # insert the MMSI number
        Day_list.insert(j, Day_list[0])
        delta_time.insert(j,delta_time[current_position]+incremental_time)
        Longitude_list.insert(j, Longitude_list[current_position])
        Latitude_list.insert(j, Latitude_list[current_position])
    return MMSI_list, Longitude_list, Latitude_list, Day_list,delta_time

def data_compensation_algorithm_movement(MMSI_list, Longitude_list, Latitude_list,
                                         Day_list, delta_time,delta_speed,
                                         number_of_compensation, current_position):
    """This function is to add some missing data.
    The processing looks like this:
    1.Whether delta time > threshold and delta < one hour,
    the missing data should be compensated.
    2.else, continue the next points.
    3.the second determines the motion patterns:
    4.if the delta V (speed increment)==0, it means the vessel is now in
    static state. The number of compensated points is n = (Tb-Ta)/Tm.
    Tm is the sample time interval. And then add n points into the lists.
    5. else the abs(delta V)>0, the vessel is in motion status.And then, add
    some points into the list."""
    # calculate the delta difference of the latitude
    increment_variable_longitude = 0.05
    increment_variable_latitude = 0.04
    increment_variable_time = 20
    for i in range(current_position, current_position + number_of_compensation):
        MMSI_list.insert(i, MMSI_list[0])
        Day_list.insert(i, Day_list[0])
        delta_time.insert(i, delta_time[i] + increment_variable_time)
        # Speed_list.insert(0,Speed_list[0])
        delta_difference_latitude = abs(Latitude_list[i + 1] - Latitude_list[i])
        # delta_difference_longitude = abs(Longitude_list[i+1]-Longitude_list[i])
        # delta_distance = math.sqrt(np.square(delta_difference_latitude)+ np.square(delta_difference_longitude))
        # if delta_speed[i] ==0:
        #     increment_variable_time = 20
        #     delta_time.insert(i, delta_time[i] + increment_variable_time)
        # else:
        #     increment_variable_time = float(delta_distance/delta_speed[i])
        #     delta_time.insert(i, delta_time[i] + increment_variable_time)
        if delta_difference_latitude > 0.05:
            # interpolation algorithm
            lat0 = Latitude_list[i]
            lng0 = Longitude_list[i]
            lat1 = Latitude_list[i + number_of_compensation]
            lng1 = Longitude_list[i + number_of_compensation]
            lat = lat0 + increment_variable_latitude
            lng = linear_interpolation_algorithm(lat0, lng0, lat1, lng1, lat)
            # insert the data into the new list
            Latitude_list.insert(i, lat)
            Longitude_list.insert(i, lng)
        else:
            Longitude_list.insert(i, Longitude_list[current_position] + increment_variable_longitude)
            Latitude_list.insert(i, Latitude_list[current_position] + increment_variable_latitude)
    return MMSI_list, Longitude_list, Latitude_list, Day_list,delta_time


def save_data_into_file(MMSI_list, Longitude_list, Latitude_list, Day_list, delta_time):
    """This function is for storing the data and outputing the data into a file."""
    # dictionary for storing the list and transfer it to DataFrame
    save_dict = {'MMSI': MMSI_list,
                 'Longitude': Longitude_list,
                 'Latitude': Latitude_list,
                 'Day': Day_list,
                 'delta_time':delta_time}
    data = pd.DataFrame(save_dict)
    # output the file
    name_mmsi = int(data.iloc[0]['MMSI'])
    name_day = int(data.iloc[0]['Day'])
    data.to_csv(str(name_mmsi)+'-'+str(name_day)+'.csv',index=False)


# load the data and calculate parameters(compensate_points, list)
'''In this part, it is about to manuplate the dataset.
Firstly, transfer the dataframe into different list (MMSI list, latitude list,
Longitude list, speed list, time list etc.).
Secondly, the loop for reading the file and calculate some parameters like
compensate_points for calculating.'''

file_address = glob.glob(r'C:\Users\LPT-ucesxc0\AIS-Data\test_data\413185000-1.csv')
threshold_time = 180  # the threshold time called 120 seconds (2 minutes)
threshold_file_size = 10250
for file in file_address:
    file_load = pd.read_csv(file)
    file_size = os.path.getsize(file)
    if file_size <= threshold_file_size:
        # transfer the DataFrame to list
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
            if delta_time[i] > threshold_time:
                for j in range(1, len(delta_speed) - 1):
                    if delta_speed[j] >= 0 and delta_speed[j] < 2.5:
                        number_of_compensation = round((time_to_seconds_list[j] -
                                                        time_to_seconds_list[j - 1]) / threshold_time)
                        current_position = j
                        new_MMSI_list, new_Longitude_list, new_Latitude_list, new_Day_list,delta_time= \
                            data_compensation_algorithm_static(MMSI_list,
                                                               Longitude_list,
                                                               Latitude_list,
                                                               Day_list,
                                                               delta_time,
                                                               number_of_compensation,
                                                               current_position)
                        save_data_into_file(new_MMSI_list,
                                            new_Longitude_list,
                                            new_Latitude_list,
                                            new_Day_list,
                                            delta_time)
                    else:
                        number_of_compensation = round((time_to_seconds_list[j] -
                                                        time_to_seconds_list[j - 1]) / threshold_time)
                        current_position = j
                        new_MMSI_list, new_Longitude_list, new_Latitude_list, new_Day_list,delta_time = \
                            data_compensation_algorithm_movement(MMSI_list,
                                                                 Longitude_list,
                                                                 Latitude_list,
                                                                 Day_list,
                                                                 delta_time,
                                                                 delta_speed,
                                                                 number_of_compensation,
                                                                 current_position)
                        save_data_into_file(new_MMSI_list,
                                            new_Longitude_list,
                                            new_Latitude_list,
                                            new_Day_list,
                                            delta_time)
            else:
                continue
    else:
        continue



