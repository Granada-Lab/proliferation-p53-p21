#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:16:08 2023

@author: nicagutu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

plt.rcParams.update({'font.size': 24})

path='Dataset/'

files = ['0G', '2G', '4G', '10G_MD']
labels = ['0Gy', '2Gy', '4Gy', '10Gy']

w = 2
before_divisions = []
after_divisions = []

for i in range(len(files)):
    data = pd.read_csv(path+'p53_'+files[i]+r'.csv', header=None, index_col=None) 
    # data2 = pd.read_csv(path+'p21_'+files[i]+r'.csv', header=0, index_col=0) 
    division_matrix = pd.read_csv(path+'division_matrix_'+files[i]+r'.csv', header=None)
    division_matrix = division_matrix.T
    division_matrix.columns = data.columns
    
    for col in data:
        division_time = (np.where(division_matrix[col]==1)[0])
        signal = data[col]
        if len(division_time)!=0 and division_time[0]>w and division_time[-1]<(len(signal)-w):
            for ii in range(len(division_time)):
                signal_before = np.mean(signal[int(division_time[ii]-w):division_time[ii]])
                signal_after = np.mean(signal[division_time[ii]:int(division_time[ii]+w)])
                before_divisions.append(signal_before)
                after_divisions.append(signal_after)
    
    corr_coefficient, _ = pearsonr(before_divisions, after_divisions)
    n = len(before_divisions)
    standard_error = np.sqrt((1-corr_coefficient**2)/int(n - 2))

print(len(before_divisions))

plt.figure(figsize=(12,10))
plt.plot(before_divisions, after_divisions, 'o', markersize=10, label=f'Correlation: {corr_coefficient:.2f} ± {standard_error:.2f}')
plt.xlabel('Mean p53 levels before division')
plt.ylabel('Mean p53 levels after division')
plt.xscale('log')
plt.yscale('log')
plt.title('all conditions')#' with a time_window: '+str(w))
plt.legend(loc='best')
plt.show()
    
before_divisions = []
after_divisions = []

for i in range(len(files)):
    data = pd.read_csv(path+'p21_'+files[i]+r'.csv', header=0, index_col=0) 
    division_matrix = pd.read_csv(path+'division_matrix_'+files[i]+r'.csv', header=None)
    division_matrix = division_matrix.T
    division_matrix.columns = data.columns
    
    for col in data:
        division_time = (np.where(division_matrix[col]==1)[0])
        signal = data[col]
        if len(division_time)!=0 and division_time[0]>w and division_time[-1]<(len(signal)-w):
            for ii in range(len(division_time)):
                signal_before = np.mean(signal[int(division_time[ii]-w):division_time[ii]])
                signal_after = np.mean(signal[division_time[ii]:int(division_time[ii]+w)])
                before_divisions.append(signal_before)
                after_divisions.append(signal_after)
    
    corr_coefficient, _ = pearsonr(before_divisions, after_divisions)
    n = len(before_divisions)
    standard_error = np.sqrt((1 - corr_coefficient**2)/int(n - 2))
                
plt.figure(figsize=(12,10))
plt.plot(before_divisions, after_divisions, 'o', markersize=10, label=f'Correlation: {corr_coefficient:.2f} ± {standard_error:.2f}')
plt.xlabel('Mean p21 levels before division')
plt.ylabel('Mean p21 levels after division')
plt.xscale('log')
plt.yscale('log')
plt.title('all conditions')#' with a time_window: '+str(w))
plt.legend(loc='best')
plt.show()
    
    