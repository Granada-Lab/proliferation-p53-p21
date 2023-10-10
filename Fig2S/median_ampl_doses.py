#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:22:55 2023

@author: nicagutu
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyboat import WAnalyzer

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'

def wavelet(signal):
    detrended_signal=wAn.sinc_detrend(signal, T_c=50)
    wAn.compute_spectrum(detrended_signal, do_plot=False) 
    wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4)
    rd=wAn.ridge_data
    return rd

path='Dataset/'
path2='Output/'

file_list=['0G', '2G', '4G', '10G_MD']
labels=['0Gy','2Gy','4Gy','10Gy']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

amplitudes_dict={}
for i in range(4):  
    file=file_list[i]
    data=pd.read_csv(path+'p53_'+file_list[i]+r'.csv',header=None) 
    time=data.index.values*0.5
    
    all_amplitudes=[]
    for column in data:
        signal = data[column]
        rd = wavelet(signal)
        all_amplitudes.append(np.median(rd['amplitude']))
        
    amplitudes_dict[i]=all_amplitudes
            
df2=pd.DataFrame.from_dict(amplitudes_dict, orient='index')
df2=df2.T
df2.columns=labels
         
fig=plt.figure(figsize=(12,10))
sns.violinplot(data=df2,showfliers=False)
sns.swarmplot(data=df2, color='black', size=1)
plt.ylabel('Median amplitude (a.u.)')
# plt.savefig(path2+'Median_period_dose.svg')
plt.show()




