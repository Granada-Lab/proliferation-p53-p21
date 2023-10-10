#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:01:20 2023

@author: nicagutu
"""

import numpy as np
from scipy.integrate import simpson
from numpy import trapz
import pandas as pd
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import seaborn as sns
from scipy.stats import pearsonr

plt.rcParams.update({'font.size': 18})
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2 = 'Output/'

files = ['0G', '2G', '4G', '10G_MD']
labels = ['0Gy', '2Gy',' 4Gy', '10Gy']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label = 'hours')

for i in range(len(files)):
    data = pd.read_csv(path+'p21_'+files[i]+r'.csv', header = 0, index_col = 0) 
    data2 = pd.read_csv(path+'p53_'+files[i]+r'.csv', header = None, index_col = None) 
    division_matrix=pd.read_csv(path+'division_matrix_'+files[i]+r'.csv', header = None)
    division_matrix=division_matrix.T
    data2.columns = data.columns
    division_matrix.columns = data.columns
    
    df = pd.DataFrame()
    for col in data:
        
        #p21 properties
        signal = data[col]
        
        trend = wAn.sinc_smooth(signal, T_c = 50)
        detrended_signal = signal-trend
        wAn.compute_spectrum(detrended_signal, do_plot = False) #computes the detrended signal wavelet spectrum
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize = 4) #gets the maximum ridge
        rd = wAn.ridge_data

        area_p21 = np.mean([trapz(signal, dx = 0.5), simpson(signal, dx = 0.5)])
        trend_p21 = np.median(trend)
        std_p21 = np.std(detrended_signal)
        amp_p21=np.median(rd['amplitude'])
        
        #p53 properties
        signal2 = data2[col]
        
        trend2 = wAn.sinc_smooth(signal, T_c=50)
        detrended_signal2 = signal2-trend2
        wAn.compute_spectrum(detrended_signal2, do_plot = False) #computes the detrended signal wavelet spectrum
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize = 4) #gets the maximum ridge
        rd2 = wAn.ridge_data

        area_p53 = np.mean([trapz(signal2, dx  =0.5), simpson(signal2, dx = 0.5)])
        trend_p53 = np.median(trend2)
        std_p53 = np.std(detrended_signal2)
        amp_p53 = np.median(rd2['amplitude'])
        per_p53 = np.median(rd2['periods'])

        #Proliferation metrics
        division_time = (np.where(division_matrix[col] == 1)[0])
        time_lastdiv = (len(signal)-division_time[-1])*0.5 if len(division_time) != 0 else 0.5*len(signal)
        imt = [(division_time[ii+1]-division_time[ii])*0.5 for ii in range(len(division_time)-1)] if len(division_time) >= 2 else [0]        
        div = len(division_time)
                
        df[col]=[area_p21, trend_p21, amp_p21, std_p21, area_p53, trend_p53, amp_p53, per_p53, std_p53, np.mean(imt), time_lastdiv, div]
    
    df = df.T
    df.columns = ['p21 AUC','p21 Trend','p21 Envelope','p21 STD','p53 AUC','p53 Trend','p53 Envelope','p53 Period','p53 STD','IMT','Cell age','Divisions']
    
    fig = plt.figure(figsize = (15,15))
    corr = df.corr()
    
    sns.heatmap(corr, annot = True, cmap = 'PRGn', cbar_kws = {'label': 'Correlation coefficient'}, vmin = -1, vmax = 1)
    
    ##Significance test
    # Generate a matrix of correlation coefficients
    corr_matrix = np.array(corr)
    N = corr_matrix.shape[0]
    corr_rand = np.random.rand(N, N)
    
    # Create a matrix to store the p-values
    pval_matrix = np.zeros_like(corr_rand)
    
    # Calculate the p-value for each correlation coefficient
    for ii in range(corr_rand.shape[0]):
        for j in range(corr_rand.shape[1]):
            if ii == j:
                pval_matrix[ii, j] = 1.0
            else:
                corr_coef, p_val = pearsonr(corr_matrix[:, ii], corr_matrix[:, j])
                pval_matrix[ii, j] = p_val
    print("P-value matrix:\n", pval_matrix)
    
    # Add asterisks to indicate the significance levels
    sig_levels = [0.001, 0.01, 0.05]
    for ii in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if ii != j:
                p_val = pval_matrix[ii, j]
                if p_val < sig_levels[0]:
                    plt.text(j + 0.5, ii + 0.5, '***', ha='center', va='bottom')
                elif p_val < sig_levels[1]:
                     plt.text(j + 0.5, ii + 0.5, '**', ha='center', va='bottom')
                elif p_val < sig_levels[2]:
                     plt.text(j + 0.5, ii + 0.5, '*', ha='center', va='bottom')
                   
    plt.title(labels[i])
    # plt.savefig(path2+'Correlation_heatmap_p21_p53_'+str(files[i])+'.svg')
    plt.show()


        
    
    