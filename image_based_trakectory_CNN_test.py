#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :Created on Dec 04 4:39 PM 2018
#@Author  :xiang chen  
import os, sys
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import pandas as pd
import glob
import datetime


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#Do no print into the scientific notation format.
np.set_printoptions(suppress=True)

#load the dataset
file_names = glob.glob(r'D:\Data store file\test-file\*.csv')
'''This loop is to address three problems:
1.Extract the time(Day,Hour,Minute,Seocond) from Record_Datetime stample.
2.Convert the Hour, Minute to seconds.
3.Add the Hour,Minute,Second together 
into a new column called Time_to_seconds.
    Args:
        Input the data files: file_names
    Retruns:
        grouped data file: groups.
     '''

# for f in file_names:
#     read_file = pd.read_csv(f)
#     #print(read_file)
#     groups = (read_file.groupby(['Day'])['MMSI',
#                                         'Longitude',
#                                         'Latitude',
#                                         'Speed',
#                                         'Day',
#                                         'Time_to_Seconds'])
#
#     for name, group in groups:
#         name_mmsi = int(group.iloc[0]['MMSI'])
#         print(name,group)
#         group.to_csv(str(name_mmsi)+'-'+str(name)+'.txt',
#                       sep='\t',index=False,header=False)

#read the ais data from the file and store it into the numpy array
file_address = glob.glob(r'D:\Data store file\test_file_numpy_array\*.txt')
for file in file_address:
    file_load = pd.read_csv(file)



def data_compensation_algorithm(ais_data,time_threshold,maximum_time_range):
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











