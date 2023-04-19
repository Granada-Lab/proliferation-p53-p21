#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 17:48:55 2023

@author: nicagutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import seaborn as sns
import scipy.stats as stats
from scipy.stats import kurtosis

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

colors_1=['tab:blue','tab:orange','tab:green','tab:red']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

for i in range(1):
    # data = pd.read_csv(path+'p53_'+file_list[i]+r'.csv', header=None) 
    data =pd.read_csv(path+'p21_'+file_list[i]+r'.csv', header = 0, index_col = 0) 
    print(len(data.columns))
    time = data.index.values*0.5
    
    properties={'amplitude': [], 'period': []}
    for column in data:
        signal = data[column]
        rd=wavelet(signal)
        
        properties['amplitude'].append(np.mean(rd['amplitude']))
        properties['period'].append(np.mean(rd['periods']))

    print('Mean', np.mean(properties['amplitude']))
    print('SD', np.std(properties['amplitude']))
    counts = np.bincount(properties['amplitude']) 
    probs = counts/len(properties['amplitude'])
    entropy = stats.entropy(probs)
    print('Entropy ', entropy)
    print("Kurtosis of the data:", kurtosis(properties['amplitude']))
    
    # N = len(data.columns)
    # mean = np.mean(properties['amplitude'])
    # sq_devs = np.sum((properties['amplitude'] - mean)**2)
    # sample_var = sq_devs / (N-1)
    # sample_std = np.sqrt(sample_var)
    # variance_of_std = ((N-1)/N) * (sq_devs - (N/(N-1)) * sample_var)
    # print('Variance of STD', variance_of_std) 
    
    fig = plt.figure(figsize = (12,10))
    sns.histplot(properties['amplitude'], kde=True, stat='density')
    plt.xlabel('Amplitude')
    # plt.savefig(path2+'Amplitude_distr_p21.svg')
    plt.show()
            
    # fig = plt.figure(figsize=(12,10))
    # sns.histplot(properties['period'], kde=True, stat='density')
    # plt.xlabel('Period [hours]')
    # # plt.savefig(path2+'Period_distr.svg')
    # plt.show()
            
    
    
    
