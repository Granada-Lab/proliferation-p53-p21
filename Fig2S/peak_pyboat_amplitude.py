#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 09:56:42 2023

@author: nicagutu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from pyboat import WAnalyzer
from scipy.stats import pearsonr

plt.rcParams.update({'font.size': 24})
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2='Output/'

file_list = ['0G','2G','4G','10G_MD','10G_NMD']
labels = ['0Gy','2Gy','4Gy','10Gy']

dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

amp_tuple_peak = {}
amp_tuple_pyboat = {}
for i in range(4):  
    file = file_list[i]    
    data = pd.read_csv(path+'p53_'+file+r'.csv', header=None) 
    time = data.index.values*0.5
    
    amplitude_all = []
    amplitude_pyboat = []
    for column in data:
        signal = data[column]
        detrended_signal = wAn.sinc_detrend(signal, T_c=50)
        wAn.compute_spectrum(detrended_signal, do_plot=False)
        rd = wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4)

        peaks, heights = find_peaks(detrended_signal, height=50, distance=7, prominence=50)
        amplitude_cell = []
        for j in range(len(heights['peak_heights'])):
            amplitude_cell.append(heights['peak_heights'][j])
        if 10 < np.median(rd['amplitude']) < 1000 and np.mean(amplitude_cell) < 1000:
            amplitude_pyboat.append(np.median(rd['amplitude']))
            amplitude_all.append(np.median(amplitude_cell))
    amp_tuple_peak[i] = amplitude_all
    amp_tuple_pyboat[i] = amplitude_pyboat
    
fig = plt.figure(figsize=(14,12))
for i in range(4):
    corr_coefficient, _ = pearsonr(amp_tuple_pyboat[i], amp_tuple_peak[i])
    n = len(amp_tuple_pyboat[i])
    standard_error = np.sqrt((1 - corr_coefficient**2)/int(n - 2))
    plt.plot(amp_tuple_pyboat[i], amp_tuple_peak[i], 'o', markersize=10, label=labels[i]+f' corr.: {corr_coefficient:.2f} ± {standard_error:.2f}')
plt.legend(loc='best')
plt.xlabel('pyBOAT median amplitude per cell')
plt.ylabel('peak-picking median amplitude per cell')
plt.xscale('log')
plt.yscale('log')
# plt.title('p53 signals')
plt.savefig(path2+'Amplitude_p53_all_conditions_peak_picking_pyboat.svg')
plt.show()    
    
# fig = plt.figure(figsize=(14,12))
# for i in range(4):
#     corr_coefficient, _ = pearsonr(amp_tuple_pyboat[i], amp_tuple_peak[i])
#     n = len(amp_tuple_pyboat[i])
#     standard_error = np.sqrt((1 - corr_coefficient**2)/int(n - 2))
#     plt.plot(np.mean(amp_tuple_pyboat[i]), np.mean(amp_tuple_peak[i]), 'o', markersize=20, label=labels[i]+f' corr.: {corr_coefficient:.2f} ± {standard_error:.2f}')
# plt.legend(loc='best')
# plt.xlabel('pyBOAT mean amplitude')
# plt.ylabel('peak-picking mean amplitude')
# plt.xscale('log')
# plt.yscale('log')
# plt.title('p53 signals')
# plt.show()    
  

                
                
                
                              
                
                
                
                