# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:24:40 2019

@author: Shiva
"""

import numpy as np
config=np.loadtxt('Line_Config.txt');
lines = np.loadtxt('edges.txt'); 
nEdges=1327
# This code is for mapping the line configuration in form of edges so that 
# voltage equations are constent
index1=[]
for k in range(0,nEdges):
    a=lines[k,0]
    b=lines[k,1]
    for m in range(0,nEdges):
        if a==config[m,0] and b==config[m,1]:
            index1.append(m)
        if a==config[m,1] and b==config[m,0]:
            index1.append(m)
            
# In the following code, the index with switches are identified such that 
# big-M method is not written for all lines.

sw = np.loadtxt('switches.txt');                        
index2=[]
for k in range(0,175):
    a=sw[k,0]
    b=sw[k,1]
    for m in range(0,nEdges):
        if a==lines[m,0] and b==lines[m,1]:
            index2.append(m)
        if a==lines[m,1] and b==lines[m,0]:
            index2.append(m)            