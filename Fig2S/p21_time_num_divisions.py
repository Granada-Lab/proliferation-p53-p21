#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:31:51 2023

@author: nicagutu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyboat import WAnalyzer

plt.rcParams.update({'font.size': 24})
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2 = 'Output/'

files = ['0G', '2G', '4G', '10G_MD']
labels = ['0Gy','2Gy','4Gy','10Gy']
colors = ['tab:blue','tab:orange','tab:green','tab:purple','tab:red','tab:brown']
divisions = [0,1,2,3,4,5]

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

for i in range(1):
    data = pd.read_csv(path+'p21_'+files[i]+r'.csv', header=0, index_col=0) 
    time = data.index.values*0.5
    division_matrix = pd.read_csv(path+'division_matrix_'+files[i]+r'.csv', header=None)
    division_matrix = division_matrix.T
    division_matrix.columns = data.columns
    
    dfs = [pd.DataFrame() for i in range(6)]
    times = {'time_1div':[], 'time_2div':[], 'time_3div':[], 'time_4div':[], 'time_5div':[]}    
    for col in data:
        signal = data[col]
        div_events = np.where(division_matrix[col] == 1)[0]
        if len(div_events) < 6:
            dfs[len(div_events)][col] = signal
            if len(div_events) > 0:
                times['time_{}div'.format(len(div_events))].append(div_events[-1]*0.5)    
                            
    fig, ax = plt.subplots(figsize=(10,8))                
    for ii in range(6):
        ax.plot(time, dfs[ii].median(axis=1), label='{} divisions'.format(ii), color=colors[ii])
        if ii > 0:
            ax.axvline(x=np.median(times['time_{}div'.format(ii)]), color=colors[i])    
    ax.set_xlabel('Time(h)')
    ax.set_ylabel('p21 levels')
    ax.legend(loc='best')
    
    ax2 = ax.twinx()
    bins_list = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for j in range(6):
        if j > 0:
            ax2.hist(times['time_{}div'.format(j)], color=colors[j], alpha=0.1, density=True, bins=bins_list, edgecolor=colors[j])    
    ax2.set_ylabel('Density time last division')
    # plt.title(str(labels[j]))
    # plt.savefig(path2+'p21_median_signal_last_division_hist.svg')
    plt.show()
    
    
    