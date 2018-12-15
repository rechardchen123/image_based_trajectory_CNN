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

# set the environments and import packages
import os, sys
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Agg') #In local debugging, it should comment it and uploading to remote server,
# should use this.
import pandas as pd
import glob
import math

# for local debugging only.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def statistic_the_data_list(plot_trajectory_dataframe):
    '''This function is used to define a clipping area. The area is a
    rectangular area.
    Input: the array of latitude and longitude
    Return: the minimum vale and maximum value both of latitude and longitude.And return
            a rectangular array'''
    get_value = []  # get the value of the results
    data_information = plot_trajectory_dataframe.describe()
    threshold_time_minimum = data_information.iloc[3]['delta_time']
    get_value.append(threshold_time_minimum)
    threshold_time_normal_navigation = data_information.iloc[4]['delta_time']
    get_value.append(threshold_time_normal_navigation)
    threshold_time_manoeuvring_navigation = data_information.iloc[6]['delta_time']
    get_value.append(threshold_time_manoeuvring_navigation)
    threshold_time_maximum = data_information.iloc[7]['delta_time']
    get_value.append(threshold_time_maximum)
    statistic_time_mean = data_information.iloc[1]['delta_time']
    get_value.append(statistic_time_mean)
    statistic_time_std = data_information.iloc[2]['delta_time']
    get_value.append(statistic_time_std)
    return get_value

# read the files.
trajectory_file_address = glob.glob(r"C:\Users\LPT-ucesxc0\AIS-Data\test_data\*.csv")
fig = plt.figure()
plt.rcParams['axes.facecolor'] = 'black' #define the backgroud
for file in trajectory_file_address:
    file_load = pd.read_csv(file)
    # get a trajectory list and transfer to array
    name_mmsi = int(file_load.iloc[0]['MMSI'])
    name_day = int(file_load.iloc[0]['Day'])
    Latitude_list = list(file_load['Latitude'])
    latitude_array = np.array(Latitude_list)
    Longitude_list = list(file_load['Longitude'])
    longitude_array = np.array(Longitude_list)
    delta_time_list = list(file_load['delta_time'])
    delta_time_array = np.array(delta_time_list)
    # the data for plot
    trajectory_lat_long_delta_time_dict = {'latitude':latitude_array,
                                           'longitude':longitude_array,
                                           'delta_time':delta_time_array}
    plot_trajectory_dataframe = pd.DataFrame(trajectory_lat_long_delta_time_dict)
    get_value = statistic_the_data_list(plot_trajectory_dataframe)
    #get the threshold time for the normal navigation:
    minimial_time = get_value[0]
    threshold_time_normal_navigation = get_value[2]
    threshold_time_maneuvring_operation = get_value[5]
    threshold_time_anchorage = 300
    maximum_time = get_value[3]

    # loop the dict
    for i in range(0,len(plot_trajectory_dataframe)-1):
        if plot_trajectory_dataframe.iloc[i]['delta_time'] == minimial_time and \
            plot_trajectory_dataframe.iloc[i]['delta_time'] <= threshold_time_normal_navigation:
            plt.plot(plot_trajectory_dataframe.iloc[i]['latitude'],
                        plot_trajectory_dataframe.iloc[i]['longitude'],
                        color='#666666',marker='*')
        elif plot_trajectory_dataframe.iloc[i]['delta_time'] > threshold_time_normal_navigation and \
            plot_trajectory_dataframe.iloc[i]['delta_time'] <= threshold_time_maneuvring_operation:
            plt.plot(plot_trajectory_dataframe.iloc[i]['latitude'],
                        plot_trajectory_dataframe.iloc[i]['longitude'],
                        color='#999999',marker='x')
        else:
            plt.plot(plot_trajectory_dataframe.iloc[i]['latitude'],
                        plot_trajectory_dataframe.iloc[i]['longitude'],
                        color='#ffffff',marker='.')
    plt.savefig(str(name_mmsi)+'-'+str(name_day)+'.jpg')
    plt.show()
    plt.close('all')





































