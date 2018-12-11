import  numpy as np
import matplotlib.pyplot as plt

#Piecewise interplot
def get_line (xn,yn):
    def line(x):
        index = -1
        # find the interval of x
        for i in range(1,len(xn)):
            if x <=xn[i]:
                index = i -1
                break
            else:
                i += 1

        if index == -1:
            return -100

        result =(x-xn[index+1])*yn[index]/float((xn[index]-xn[index+1]))+\
                (x-xn[index])*yn[index+1]/float((xn[index+1]-xn[index]))
        return result
    return line

xn = [i for i in range(-50,50,10)]
yn = [i**2 for i in xn]

lin = get_line(xn,yn)

x = [i for i in range(-50,40)]
y = [lin(i) for i in x]

plt.plot(xn,yn,'ro')
plt.plot(x,y,'b-')
plt.show()
