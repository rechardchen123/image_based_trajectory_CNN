#!/usr/bin/env python3
# _*_coding:utf-8 _*_
# @Time    :Created on Dec 04 4:39 PM 2018
# @Author  :xiang chen
import os
import pandas as pd
import numpy as np
import glob
import math

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

def linear_interpolation_algorithm(lat0, lng0, lat1, lng1, lat):
    """Algorithm for liner interpolation
    Input: The input number
    Output: The interpolation value."""
    delta = (lng1 - lng0) / (lat1 - lat0)
    lng = lng0 + delta * (lat - lat0)
    return lng


def data_compensation_algorithm_static(MMSI_list,Longitude_list,Latitude_list,Speed_list,
                                       Heading_list,Day_list,time_to_seconds_list,delta_time,
                                       delta_speed,delta_heading,number_of_compensation,
                                       current_position):
    """The function is about to add new static points into the trajectory sequences.
    All the list should be added.
    Input parameters: the original list
    Output: the new list contained the added points."""
    incremental_time = 30 # for static traget, the time should be reported to station between 30s.
    for j in range(current_position, current_position + number_of_compensation):
        MMSI_list.insert(j, MMSI_list[0])  # insert the MMSI number
        Longitude_list.insert(j, Longitude_list[current_position])
        Latitude_list.insert(j, Latitude_list[current_position])
        Speed_list.insert(j, Speed_list[current_position] + incrementtal_speed)
        Heading_list.insert(j, Heading_list[j])
        Day_list.insert(j, Day_list[0])
        time_to_seconds_list(j,time_to_seconds_list[j-1]+incremental_time)
        delta_time.insert(j,delta_time[current_position]+incremental_time)
        delta_speed.insert(j,delta_speed[j])
        delta_heading.insert(j,delta_heading[j])
    return MMSI_list, Longitude_list, Latitude_list, Speed_list,Heading_list,Day_list,\
           time_to_seconds_list,delta_time,delta_speed,delta_heading

def data_compensation_algorithm_movement(MMSI_list, Longitude_list, Latitude_list,Speed_list,
                                         Heading_list, Day_list, time_to_seconds_list, delta_time,
                                         delta_speed, delta_heading, number_of_compensation,
                                         current_position):
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
        delta_difference_latitude = abs(Latitude_list[i + 1] - Latitude_list[i])
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
        Speed_list.insert(i,Speed_list[i-1]+delta_speed[i-1])
        Heading_list.insert(i,Heading_list[i-1]+delta_heading[i-1])
        Day_list.insert(i, Day_list[0])
        time_to_seconds_list.insert(j,time_to_seconds_list[j-1]+incremental_time)
        delta_time.insert(i, delta_time[i] + increment_variable_time)
        delta_speed.insert(i,abs(Speed_list[i]-Speed_list[i-1]))
        delta_heading.insert(i,abs(Heading_list[i]-Heading_list[i-1]))
    return MMSI_list, Longitude_list, Latitude_list, Speed_list,Heading_list,Day_list,\
           time_to_seconds_list,delta_time,delta_speed,delta_heading


def save_data_into_file(MMSI_list, Longitude_list, Latitude_list,Speed_list,
                        Heading_list,Day_list,time_to_seconds_list,delta_time,
                        delta_speed,delta_heading):
    """This function is for storing the data and outputing the data into a file."""
    # dictionary for storing the list and transfer it to DataFrame
    save_dict = {'MMSI': MMSI_list,
                 'Longitude': Longitude_list,
                 'Latitude': Latitude_list,
                 'Speed':Speed_list,
                 'Heading':Heading_list,
                 'Day': Day_list,
                 'time_to_seconds':time_to_seconds_list,
                 'delta_time':delta_time,
                 'delta_speed':delta_speed,
                 'delta_heading':delta_heading}
    data = pd.DataFrame(save_dict)
    # output the file
    name_mmsi = int(data.iloc[0]['MMSI'])
    name_day = int(data.iloc[0]['Day'])
    data.to_csv('/home/ucesxc0/Scratch/output/ais_trajectory_compensation/result/%d-%d.csv' % (name_mmsi, name_day),
                index=False)


# load the data and calculate parameters(compensate_points, list)
'''In this part, it is about to manipulate the dataset.
Firstly, transfer the DataFrame into different list (MMSI list, latitude list,
Longitude list, speed list, time list etc.).
Secondly, the loop for reading the file and calculate some parameters like
compensate_points for calculating.'''

file_address = glob.glob('/home/ucesxc0/Scratch/output/ais_trajectory_compensation/AIS-data-after-day-split/*.csv')
threshold_time = 120  # the threshold time called 120 seconds (2 minutes)
threshold_waiting_time = 400
threshold_speed = 2
threshold_heading = 6
threshold_heading_max = 20
threshold_file_size = 20*1024
for file in file_address:
    file_load = pd.read_csv(file)
    # transfer the DataFrame to list
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

    # calculate the number of compensation points
    for i in range(1,len(Speed_list)-1):
        if Speed_list[i] < threshold_speed and delta_time[i]>threshold_waiting_time:
                    number_of_compensation = round(delta_time[i]/threshold_waiting_time)
                    current_position = i
                    new_MMSI_list,new_Longitude_list,new_Latitude_list,new_Speed_list,new_Heading_list,\
                    new_Day_list,new_time_to_seconds_list,new_delta_time,new_delta_speed,new_delta_heading=\
                        data_compensation_algorithm_static(MMSI_list,Longitude_list,Latitude_list,
                                                           Speed_list,Heading_list,Day_list,time_to_seconds_list,
                                                           delta_time,delta_speed,delta_heading,number_of_compensation,
                                                           current_position)
                    save_data_into_file(new_MMSI_list,new_Longitude_list,new_Latitude_list,
                                        new_Speed_list,new_Heading_list,new_Day_list,
                                        new_time_to_seconds_list,new_delta_time,
                                        new_delta_speed,new_Heading_list)
        elif (Speed_list[i]>=threshold_speed and delta_time[i]>threshold_time) or (delta_heading > threshold_heading
                                and delta_heading<threshold_heading_max):
            number_of_compensation = round(delta_time[i]/threshold_time)
            current_position = i
            new_MMSI_list, new_Longitude_list, new_Latitude_list, new_Speed_list, new_Heading_list, \
            new_Day_list, new_time_to_seconds_list, new_delta_time, new_delta_speed, new_delta_heading =\
                data_compensation_algorithm_movement(MMSI_list,Longitude_list,Latitude_list,
                                                           Speed_list,Heading_list,Day_list,time_to_seconds_list,
                                                           delta_time,delta_speed,delta_heading,number_of_compensation,
                                                           current_position)
            save_data_into_file(new_MMSI_list,new_Longitude_list,new_Latitude_list, new_Speed_list, new_Heading_list, \
            new_Day_list, new_time_to_seconds_list, new_delta_time, new_delta_speed, new_delta_heading)
