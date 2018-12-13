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
#matplotlib.use('Agg') #In local debugging, it should comment it and uploading to remote server,
                      # should use this.
import pandas as pd
import glob
import math
# for local debugging only.
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def Calculate_the_centorid_of_trajectory(latitude_array,longitude_array):
    """The function is used to calculate the centorid of the trajectories.
    Input:the array of latitude and longitude
    Return:the mean value of the two arraies."""
    latMean = np.mean(latitude_array)
    longMean = np.mean(longitude_array)
    return [latMean,longMean]

def Clip_area(latitude_array,longitude_array):
    '''This function is used to define a clipping area. The area is a
    rectangular area.
    Input: the array of latitude and longitude
    Return: the minimum vale and maximum value both of latitude and longitude.And return
            a rectangular array'''
    minimum_latitude = latitude_array.min()
    minimum_longitude = longitude_array.min()
    maximum_latitude = latitude_array.max()
    maximum_longitude = longitude_array.max()
    # get the clipping area
    x_coordinate = [minimum_latitude,maximum_latitude]
    y_coordinate = [minimum_longitude,maximum_longitude]
    return x_coordinate, y_coordinate

def plot_the_trajectory_graph(latitude_array, longitude_array,name_mmsi,name_day,
                              ):
    """This function is used to produce the orgign trajectory"""
    file_name_mmsi = name_mmsi
    file_name_day = name_day
    x_row = latitude_array
    y_column = longitude_array
    plt.rcParams['axes.facecolor'] = 'black'
    plt.plot(x_row,y_column,'w')
    plt.xticks(np.arange(min(x_row),max(x_row),0.002))
    plt.yticks(np.arange(min(y_column),max(y_column),0.004))
    #plt.grid(True)
    plt.savefig(r"C:\Users\LPT-ucesxc0\AIS-Data\trajectory_befor_clip\%d-%d.jpg" % (name_mmsi, name_day))
    #plt.cla()
    # in remote server, should comment on the show command.
    plt.show()

# read the files.
trajectory_file_address = glob.glob(r"C:\Users\LPT-ucesxc0\AIS-Data\test_data\*.csv")
for file in trajectory_file_address:
    file_load = pd.read_csv(file)
    #get a trajectory list and transfer to array
    name_mmsi = file_load.iloc[0]['MMSI']
    name_day = file_load.iloc[0]['Day']
    Latitude_list = list(file_load['Latitude'])
    latitude_array = np.array(Latitude_list)
    Longitude_list = list(file_load['Longitude'])
    longitude_array = np.array(Longitude_list)
    delta_time_list = list(file_load['delta_time'])
    delta_time_array = np.array(delta_time_list)
    #get the centorid of the arraies
    latMean, longMean = Calculate_the_centorid_of_trajectory(latitude_array,longitude_array)
    print(latMean,longMean)
    # get the range of clip area
    x_coordinate, y_coordinate = Clip_area(latitude_array,longitude_array)
    print(x_coordinate,y_coordinate)
    plot_the_trajectory_graph(latitude_array,longitude_array,name_mmsi,name_day)






