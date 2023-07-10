#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 13:58:13 2022

@author: nicagutu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import seaborn as sns

plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2 = 'Output/'
file_list = ['0G', '2G', '4G', '10G_MD']
labels = ['0Gy','2Gy','4Gy','10Gy']

num_pulses = {}
for i in range(4):  
    file = file_list[i]    
    data = pd.read_csv(path+'p53_'+file+r'.csv',header=None) 
    time = data.index.values*0.5
    # print(data)    
    
    pulses = []
    for column in data:
        signal = data[column]
        peaks,heights = find_peaks(signal, height=50, distance=8, prominence=50)
        if len(peaks)>2:
            pulses.append(len(peaks))
    num_pulses[i] = pulses
    
df = pd.DataFrame.from_dict(num_pulses, orient='index')
df = df.T
df.columns = labels
# print(df)   

fig = plt.figure(figsize=(10,8))
sns.violinplot(data=df)
plt.ylabel('Number of pulses')
plt.savefig(path2+'Number_pulses.svg')
plt.show()

