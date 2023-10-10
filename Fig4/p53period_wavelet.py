#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:55:22 2023

@author: nicagutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
from pyboat import ensemble_measures as em

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'      

path='Data/'
path2='Output/'

file_list = ['0G','2G','4G','10G_MD','10G_NMD']
labels = ['0Gy','2Gy','4Gy','10Gy']

colors_1=['tab:blue','tab:orange','tab:green','tab:red']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label = 'hours')

fig = plt.figure(figsize = (12,10))
for i in range(4):
    data = pd.read_csv(path+'p53_'+file_list[i]+r'.csv', header = None) 
    time = data.index.values*0.5
    
    ridge_results = {}
    
    for column in data:
        signal = data[column]
        
        detrended_signal = wAn.sinc_detrend(signal, T_c = 50)
        norm_amp = wAn.normalize_amplitude(detrended_signal, window_size=50)
        wAn.compute_spectrum(norm_amp, do_plot = False) 
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize = 4) #gets the maximum ridge
        rd = wAn.ridge_data
        ridge_results[column] = rd
    
    powers_series = em.average_power_distribution(ridge_results.values(),signal_ids = ridge_results.keys())
    high_power_ids = powers_series[powers_series > 1].index
    high_power_ridge_results = [ridge_results[i] for i in high_power_ids]
    res = em.get_ensemble_dynamics(high_power_ridge_results)
    
    plt.plot(time,res[0]['median'],label=labels[i],color=colors_1[i])
    plt.fill_between(time, res[0]['Q1'], res[0]['Q3'], color=colors_1[i], alpha=.1)
    
plt.ylabel('Period (h)')
plt.xlabel('Time (h)')
plt.legend(loc = 'best')
# plt.savefig(path2+'Wavelet_analysis_period.svg')
plt.show()

