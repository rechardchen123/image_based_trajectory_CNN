import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#read the file
data = pd.read_csv('412209050-24-test.csv')
print(data)
Longitude_list = list(data['Longitude'])
Latitude_list = list(data['Latitude'])

#plot the image
plt.figure(figsize=(8,5),dpi=80)
plt.subplot(111)
plt.plot(Latitude_list,Longitude_list)
plt.grid(True,color='r',linestyle='-')

# plt.xlim(min(Latitude_list) * 1.1, max(Latitude_list) * 1.1)
# plt.ylim(min(Longitude_list) * 1.1, max(Longitude_list) * 1.1)
plt.show()
