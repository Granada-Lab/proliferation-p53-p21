#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:35:34 2023

@author: nicagutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import seaborn as sns
from scipy.stats import ttest_ind

plt.rcParams.update({'font.size': 22})  
plt.rcParams['svg.fonttype'] = 'none'  

path = 'Datatset/'
path2 = 'Output/'

file_list = ['0G', '2G', '4G', '10G_MD']
labels = ['0Gy', '2Gy', '4Gy', '10Gy']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2 
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label = 'hours')

for i in range(1):
    file = file_list[i]
    
    data = pd.read_csv(path+'p53_'+file+r'.csv', header = None) 
    time = data.index.values*0.5
    
    division_matrix = pd.read_csv(path+'division_matrix_'+file_list[i]+r'.csv', header = None)
    division_matrix = division_matrix.T
    
    prolif={'low_prolif': [], 'high_prolif': []}
    for column in data:
        signal = data[column]
        
        detrended_signal = wAn.sinc_detrend(signal, T_c = 50)
        norm_amp = wAn.normalize_amplitude(detrended_signal, window_size=50)
        wAn.compute_spectrum(norm_amp, do_plot = False) 
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize =4 ) 
        rd=wAn.ridge_data

        prolif['low_prolif' if len(np.where(division_matrix[column] == 1)[0]) <= 1 else 'high_prolif'].append(np.median(rd['periods']))

    df_list=pd.DataFrame().from_dict(prolif, orient='index')
    df_list=df_list.T
    df_list.columns=['Low proliferative','High proliferative']
    
    fig=plt.figure(figsize = (12,10))
    sns.violinplot(data=df_list,showfliers=False)
    t_stat, p_value = ttest_ind(df_list['Low proliferative'].dropna(), df_list['High proliferative'].dropna(), equal_var=False)
    plt.text(0.5, 10, f" {p_value:.2e}", fontsize=12, ha='center', va='center')

    plt.ylabel('Median period (h)')
    plt.savefig(path2+'Median_period_low_high_prolif_'+str(labels[i])+'.svg')
    plt.show()
    

