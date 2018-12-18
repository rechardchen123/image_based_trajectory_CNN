#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/18/2018 8:01 PM 
# @Author : Xiang Chen (Richard)
# @File : test.py 
# @Software: PyCharm
import numpy as np
import pandas as pd

x = {'x1':[1,2,3,4,5,6,7],
     'y1':[2,12,34,55,67,45,33]}

x1 = [1,2,3,4,5,6,7]
y1 = [2,12,34,55,67,45,33]
l = [i for i in range(x1)]
n=3
print([l[i:i+n] for i in range(0,len(l),n)])





