import os,sys
import glob
import pandas as pd

file_address = glob.glob(r'C:\Users\LPT-ucesxc0\AIS-Data\test_data\*.csv')
threshold_time = 300  # the threshold time called 120 seconds (2 minutes)
for file in file_address:
    file_load = pd.read_csv(file)
    file_size = os.path.getsize(file)
    print(file_size)