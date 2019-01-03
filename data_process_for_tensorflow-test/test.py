#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 12/29/2018 6:39 PM 
# @Author : Xiang Chen (Richard)
# @File : test.py 
# @Software: PyCharm
def f(x,y,z,n):
    return x,y,z,n
l1 = [1,2,3,4,5]
l2 = ['x','y','z','m','n']
l3 = ['x1','y1','z1','m1','n1']
l4 = ['x11','y11','z11','m11','n11']
print(list(map(f,l1,l2,l3,l4)))
