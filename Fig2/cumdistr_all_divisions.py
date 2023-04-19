#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 14:39:11 2023

@author: nicagutu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2 = 'Output/'

files = ['0G', '2G', '4G', '10G_MD']
labels = ['0Gy','2Gy','4Gy','10Gy']

data = pd.read_csv(path+'division_matrix_'+str(files[0])+r'.csv')
data = data.T
time = np.arange(0,len(data.index.values),step = 1)

division_profile = (data.to_numpy())
division_profile = division_profile.T

distr = np.sum(division_profile[:len(data.columns), :], axis = 0)
maximum = max(np.cumsum(distr))   

size_0G=len(data.columns)

fig = plt.figure(figsize = (12,10))   
for i in range(len(files)):
    file='division_matrix_'+str(files[i])

    data = pd.read_csv(path+file+r'.csv')
    data = data.T
    
    median_distr = pd.DataFrame()
    for a in range(100):
        if i == 0:
            num_Ids = np.arange(0,size_0G,step = 1)
        else:
            num_Ids = random.sample(list(data.columns), size_0G)
            
        time_points = len(data.index.values)
        division_profile = np.zeros((len(num_Ids),time_points))
        
        division_profile = np.array([data[col].values for col in num_Ids])
                        
        distr = np.sum(division_profile[:len(data.columns), :], axis=0)
        median_distr[a] = distr/maximum
    
    average_distr = median_distr.mean(axis=1)
    std = median_distr.std(axis=1)
    
    plt.errorbar(time,np.cumsum(average_distr),yerr = std/np.sqrt(size_0G),label = str(labels[i]))

plt.xlabel('Time(h)')
plt.ylabel('Cumulative distribution of division events')
plt.legend(loc = 'best')
# plt.savefig(path2+'Cumul_distr_all_divisions.svg')
plt.show()




