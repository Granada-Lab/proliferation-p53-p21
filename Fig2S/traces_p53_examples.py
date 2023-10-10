#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:53:43 2023

@author: nicagutu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'      

path = 'Datatset/'
path2 = 'Output/'

files=['0G', '2G', '4G', '10G_MD']
labels=['0Gy','2Gy','4Gy','10Gy']
divisions=[0, 1, 2, 3, 4, 5]

for i in range(2,3):
    data = pd.read_csv(path+'p53_'+files[i]+r'.csv', header = None, index_col = None) 
    division_matrix = pd.read_csv(path+'division_matrix_'+files[i]+r'.csv', header = None)
    division_matrix = division_matrix.T
    time = data.index.values*0.5
    print(files[i])
    
    count = 0
    for col in data:
        signal = data[col]
        division_times = (np.where(division_matrix[col] == 1)[0])
        rand = np.random.uniform(0,1)
        
        if rand<=0.5 and count<=10:
            count+=1
        
            plt.figure(figsize=(8,6))
            plt.plot(time, signal, linewidth=4)
            for ii in range(len(division_times)):
                plt.axvline(x=division_times[ii]*0.5, color='black')
            plt.xlabel('Time [hours]')
            plt.ylabel('p53 levels [a.u.]')
            plt.savefig(path2+files[i]+'_p53_signal_'+str(count+10)+'.svg')
            plt.show()
            
            
    
