#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/19/2018 7:54 PM 
# @Author : Xiang Chen (Richard)
# @File : trajectory_generation_local_new.py 
# @Software: PyCharm
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

trajectory_file_address = glob.glob(r"C:\Users\LPT-ucesxc0\AIS-Data\test_data\*.csv")
plt.rcParams['axes.facecolor'] = 'black'  # define the backgroud

for file in trajectory_file_address:
    file_load = pd.read_csv(file)
    # get a trajectory list and transfer to array
    name_mmsi = int(file_load.iloc[0]['MMSI'])
    longitude_list = list(file_load['Longitude'])
    latitude_list = list(file_load['Latitude'])
    speed_list = list(file_load['Speed'])
    heading_list = list(file_load['Heading'])
    name_day = int(file_load.iloc[0]['Day'])
    delta_time_list = list(file_load['delta_time'])
    delta_speed_list = list(file_load['delta_speed'])
    delta_heading_list = list(file_load['delta_heading'])
    # the data for plot
    trajectory_lat_long_speed_heading_delta_time_speed_heading_dict = {
        'latitude': latitude_list,
        'longitude': longitude_list,
        'speed': speed_list,
        'heading': heading_list,
        'delta_time': delta_time_list,
        'delta_speed': delta_speed_list,
        'delta_heading': delta_heading_list
    }
    plot_trajectory_dataframe = pd.DataFrame(trajectory_lat_long_speed_heading_delta_time_speed_heading_dict)
    speed_threshold = 2.0
    delta_heading_threshold = 8
    #get the deviation
    speed_deviation = plot_trajectory_dataframe['speed'].std()
    delta_heading_max = plot_trajectory_dataframe['delta_heading'].max()
    #loop for the file
    for i in range(1, len(plot_trajectory_dataframe)):
        if plot_trajectory_dataframe.iloc[i]['speed'] <= speed_threshold:
            plt.plot(plot_trajectory_dataframe.iloc[i]['latitude'],
                     plot_trajectory_dataframe.iloc[i]['longitude'],
                     color='#ffffff', marker='.')  # berthing or anchorage
        else:
            if plot_trajectory_dataframe.iloc[i]['delta_heading'] <= delta_heading_threshold:
                plt.plot(plot_trajectory_dataframe.iloc[i]['latitude'],
                         plot_trajectory_dataframe.iloc[i]['longitude'],
                         color='#c0c0c0', marker='.')  # normal navigation
            else:
                plt.plot(plot_trajectory_dataframe.iloc[i]['latitude'],
                         plot_trajectory_dataframe.iloc[i]['longitude'],
                         color='#666666', marker='.')  # maneuvring operation
    #label for the trajectory image
    if speed_deviation <2.0:
        name_label_static = 0
        plt.savefig(r'C:\Users\LPT-ucesxc0\Documents\Github-repositories\image_based_trajectory_CNN/%d-%d-%d.jpg' % (
        name_mmsi, name_day, name_label_static))
        plt.show()
        plt.close('all')
    elif delta_heading_max <=delta_heading_threshold:
        name_label_normal_navigation = '0-1'
        plt.savefig(r'C:\Users\LPT-ucesxc0\Documents\Github-repositories\image_based_trajectory_CNN/%d-%d-%s.jpg' % (
            name_mmsi, name_day, name_label_normal_navigation))
        plt.show()
        plt.close('all')
    else:
        name_label_maneuvring = '0-1-2'
        plt.savefig(r'C:\Users\LPT-ucesxc0\Documents\Github-repositories\image_based_trajectory_CNN/%d-%d-%s.jpg' % (
            name_mmsi, name_day, name_label_maneuvring))
        plt.show()
        plt.close('all')


