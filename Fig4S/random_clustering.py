#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 14:58:33 2023

@author: nicagutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import random

plt.rcParams.update({'font.size': 22})    
plt.rcParams['svg.fonttype'] = 'none'

def wavelet(signal):
    detrended_signal=wAn.sinc_detrend(signal, T_c = 50)
    wAn.compute_spectrum(detrended_signal, do_plot=False) #computes the detrended signal wavelet spectrum
    wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) #gets the maximum ridge
    rd=wAn.ridge_data
    return rd

path = 'Dataset/'
path2 = 'Output/'

file_list = ['0G','2G','4G','10G_MD']
labels = ['0Gy','2Gy','4Gy','10Gy']

colors = ['black','tab:blue','tab:green','tab:purple']
colors2 = ['black','blue','green','purple']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

df_per1 = pd.DataFrame()
df_per2 = pd.DataFrame()
cum_distr1 = pd.DataFrame()
cum_distr2 = pd.DataFrame()

for i in range(1,4):
    data = pd.read_csv(path+'p53_'+file_list[i]+r'.csv', header = None) 
    divisions = pd.read_csv(path+'division_matrix_'+file_list[i]+r'.csv', header = None)
    divisions = divisions.T
    divisions.columns = data.columns
    time = data.index.values*0.5
    
    sample_size = 600 
    median_distr1 = pd.DataFrame()
    median_distr2 = pd.DataFrame()
    median_period1 = pd.DataFrame()
    median_period2 = pd.DataFrame()
    for a in range(20):
        IDs=random.sample(list(data.columns),sample_size)
        IDs1=IDs[0:int(sample_size/2)]    
        IDs2=IDs[int(sample_size/2):sample_size]    
                        
        periods1 = ([wavelet(data[col])['periods'] for col in IDs1])
        median_period1[a] = np.median(periods1, axis=0)
    
        periods2 = ([wavelet(data[col])['periods'] for col in IDs2])
        median_period2[a] = np.median(periods2, axis=0)
                         
        #Cumulative distribution           
        division_profile1 = np.array([divisions[int(j)] for j in IDs1])                    
        distr1 = np.sum(division_profile1[:len(IDs1), :], axis=0)
            
        division_profile2 = np.array([divisions[int(j)] for j in IDs2])                    
        distr2 = np.sum(division_profile2[:len(IDs2), :], axis=0)
        
        maximum = max(max(np.cumsum(distr1)),max(np.cumsum(distr2)))
        
        median_distr1[a] = np.cumsum(distr1/(maximum))
        median_distr2[a] = np.cumsum(distr2/(maximum))

    df_per1[labels[i]] = median_period1.median(axis=1)
    df_per2[labels[i]] = median_period2.median(axis=1)

    cum_distr1[labels[i]] = median_distr1.median(axis=1)
    cum_distr2[labels[i]] = median_distr2.median(axis=1)
    
df_per1.to_csv(path+'Random_cluster1_periods.csv')
df_per2.to_csv(path+'Random_cluster2_periods.csv')
cum_distr1.to_csv(path+'Random_cluster1_cumdistr.csv')
cum_distr2.to_csv(path+'Random_cluster2_cumdistr.csv')

fig = plt.figure(figsize = (12,10))
for i in range(1,4):
    plt.plot(time,df_per1[labels[i]],label=str(labels[i])+' g1',color=colors[i])
    plt.plot(time,df_per2[labels[i]],label=str(labels[i])+' g2',color=colors2[i])
plt.xlabel('Time [h]')
plt.ylabel('Period [h]')
plt.legend(loc='best')
plt.savefig(path2+'Null_hypothesis_periods_random_class.svg')
plt.show()

fig = plt.figure(figsize = (12,10))
for i in range(1,4):
    plt.plot(time,cum_distr1[labels[i]],label=str(labels[i])+' g1',color=colors[i])
    plt.plot(time,cum_distr2[labels[i]],label=str(labels[i])+' g2',color=colors2[i])
plt.xlabel('Time(h)')
plt.ylabel('Cumulative distribution of division events')
plt.legend(loc='best')
plt.savefig(path2+'Null_hypothesis_cumdistr_random_class.svg')
plt.show()

