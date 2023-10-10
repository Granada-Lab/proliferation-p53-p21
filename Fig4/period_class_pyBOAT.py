#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:24:29 2023

@author: nicagutu
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyboat import WAnalyzer
from pyboat import ensemble_measures as em
import seaborn as sns
from collections import Counter

plt.rcParams.update({'font.size': 24})
plt.rcParams['svg.fonttype'] = 'none'      

def linear(a,b,x):
    return (a+b*x)

path = 'Datatset/'
path2 = 'Output/'

files = ['2G', '4G', '10G_MD']
labels = ['2Gy', '4Gy', '10Gy']

dt = 0.5 
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label = 'hours')

df_fraction = {}
frac_stables = []
frac_prolong = []
all_cells=2473 #only damaged cells
for i in range(3):
    data = pd.read_csv(path+'p53_'+files[i]+r'.csv', header = None)
    time = data.index.values
    division_matrix = pd.read_csv(path+'division_matrix_'+files[i]+r'.csv',header = None)
    division_matrix = division_matrix.T
    division_matrix.columns = data.columns
    
    ridge_results_1 = {}
    ridge_results_2 = {}
    values=[]
    df_const_period=pd.DataFrame()
    df_incr_period=pd.DataFrame()
    for column in data: 
        signal=data[column] 
        detrended_signal = wAn.sinc_detrend(signal, T_c = 50)
        norm_amp = wAn.normalize_amplitude(detrended_signal, window_size=50)
        wAn.compute_spectrum(norm_amp, do_plot = False) 
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize = 4) #gets the maximum ridge
        rd = wAn.ridge_data
        
        diff = (np.mean(rd['periods'][0:40])-np.mean(rd['periods'][200:240]))
        trend = wAn.sinc_smooth(rd['periods'], T_c = 50)
        popt = np.polyfit(time,trend,1)

        if popt[0]>0 and popt[0]<0.01:
            ridge_results_1[column] = rd
            values.append(0)
            df_const_period[column] = rd['periods']
            
        elif popt[0] >= 0.01 and diff < 0:
            ridge_results_2[column] = rd
            values.append(1)
            df_incr_period[column] = rd['periods']
            
    df_fraction[i] = values
    frac_stables.append(Counter(values)[0]/len(data.columns)) #all_cells
    frac_prolong.append(Counter(values)[1]/len(data.columns)) #all_cells
    
    df_incr_period.to_csv(path+'Incrperiod_'+str(files[i])+'.csv')  
    
    powers_series = em.average_power_distribution(ridge_results_1.values(),signal_ids = ridge_results_1.keys())
    high_power_ids = powers_series[powers_series > 1].index
    high_power_ridge_results = [ridge_results_1[i] for i in high_power_ids]
    res = em.get_ensemble_dynamics(high_power_ridge_results)
    
    df_const_period.to_csv(path+'Constperiod_'+str(files[i])+'.csv')  
    
    powers_series = em.average_power_distribution(ridge_results_2.values(),signal_ids = ridge_results_2.keys())
    high_power_ids = powers_series[powers_series > 1].index
    high_power_ridge_results = [ridge_results_2[i] for i in high_power_ids]
    res1 = em.get_ensemble_dynamics(high_power_ridge_results)
    
    fig = plt.figure(figsize = (12,10))
    plt.plot(time*0.5,res[0]['median'],label='Constant',color='tab:blue')
    plt.fill_between(time*0.5, res[0]['Q1'], res[0]['Q3'], color='tab:blue', alpha=.1)
    plt.plot(time*0.5,res1[0]['median'],'--',label='Increasing',color='tab:green')
    plt.fill_between(time*0.5, res1[0]['Q1'], res1[0]['Q3'], color='tab:green', alpha=.1)
    plt.ylabel('Period (h)')
    plt.xlabel('Time (h)')
    plt.legend(loc='best')
    plt.title(str(labels[i]))
    plt.show()

df_all=pd.DataFrame().from_dict(df_fraction, orient='index')
df_all=df_all.T
df_all.columns=labels

fig = plt.figure(figsize = (12,10))
sns.barplot(data=df_all)
plt.ylabel('Fraction of cells with increasing period')
plt.ylim([0,1])
plt.show()

fig = plt.figure(figsize = (12,10))
plt.bar(labels,frac_stables,label='stables',color='tab:blue')
plt.bar(labels,frac_prolong,bottom=frac_stables,label='prolongers',color='tab:orange')
plt.legend(loc='best')
plt.ylabel('Fraction of cells')
plt.ylim([0,1])
plt.show()

print(frac_stables)
print(frac_prolong)
print(all_cells)

