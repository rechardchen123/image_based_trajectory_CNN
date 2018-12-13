import matplotlib.pyplot as plt
import numpy as np
x =[0,5,9,10,15]
y = [0,1,2,3,4]
plt.rcParams['axes.facecolor'] = 'black'
plt.figure()
#plt.plot(x,y,'wo',linewidth=1)
plt.plot(x,y,'#292421')
plt.xticks(np.arange(min(x),max(x)+1,1.0))
plt.show()
