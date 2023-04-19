#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:23:37 2023

@author: nicagutu
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
from scipy.stats import ttest_ind

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2 = 'Output/'

files = ['0G', '2G', '4G', '10G_MD']
labels = ['0Gy','2Gy','4Gy','10Gy']
colors = ['tab:blue','tab:orange','tab:green','tab:purple']
max_divs = [5, 6, 6, 4]
#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT=10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

df=pd.DataFrame()
for i in range(3,4):
    print(files[i])
    data = pd.read_csv(path+'p21_'+files[i]+r'.csv', header = 0, index_col = 0) 
    data2 = pd.read_csv(path+'p53_'+files[i]+r'.csv', header = None, index_col = None) 
    division_matrix=pd.read_csv(path+'division_matrix_'+files[i]+r'.csv', header = None)
    division_matrix=division_matrix.T
    data2.columns = data.columns
    division_matrix.columns = data.columns
    
    max_div = int(max_divs[i])+1
    amp_p53 = {}
    amp_p21 = {}
    
    for a in max_divs:
        amp_p53.update({k: [] for k in range(max_div)})
        amp_p21.update({k: [] for k in range(max_div)})
        
    for column in data:
        signalp53 = data2[column]
        detrended_signalp53 = wAn.sinc_detrend(signalp53, T_c = 50)
        wAn.compute_spectrum(detrended_signalp53, do_plot = False) 
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize = 4) 
        rdp53 = wAn.ridge_data

        signalp21 = data[column]
        detrended_signalp21 = wAn.sinc_detrend(signalp21, T_c = 50)
        wAn.compute_spectrum(detrended_signalp21, do_plot = False) 
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize = 4) 
        rdp21 = wAn.ridge_data

        division_times = (np.where(division_matrix[column] == 1)[0])
        num_div = len(division_times)
        amp_p53[num_div].append(np.mean(rdp53['amplitude']))
        amp_p21[num_div].append(np.mean(rdp21['amplitude']))#np.std(detrended_signalp21)) #or amplitude of p21
                       
    df_list = pd.DataFrame().from_dict(amp_p53, orient='index')
    df_list = df_list.T
    upper_limit = df_list[0].quantile(0.75) + 1.5 * (df_list[0].quantile(0.75) - df_list[0].quantile(0.25))
    high = df_list[0][df_list[0] <= upper_limit].max()    
    medians = df_list.median()
    corr = np.corrcoef(np.arange(0, max_div, step=1), medians)[0,1]
    se = np.sqrt((1 - corr**2) / (len(medians) - 2))
    # print(corr, se)
    p_values_matrix = np.zeros((max_div, max_div))      
    fig=plt.figure(figsize = (12,10))
    df_list.boxplot(showfliers = False)
    for jj in range(max_div):
        for ii in range(max_div):
            if ii > jj:
                t_stat, p_value = ttest_ind(df_list[jj].dropna(), df_list[ii].dropna(), equal_var=False)
                print("P-value of "+str(jj)+" to "+str(ii)+":", p_value)
                p_values_matrix[jj,ii] = p_value
                if jj+1 == ii:
                    t_stat, p_value = ttest_ind(df_list[jj].dropna(), df_list[ii].dropna(), equal_var=False)
                    plt.text(1.5+1*jj, high, f" {p_value:.2e}", fontsize=12, ha='center', va='center')
    plt.text(max_div-2, high*0.75, r'Corr coeff: {:.2f} $\pm$ {:.2f}'.format(corr,se))
    plt.ylabel('p53 amplitude (a.u.)')
    plt.xlabel('Number divisions')
    # plt.savefig(path2+'Boxplot_ampl_p53_divisions_'+str(labels[i])+'.svg')
    plt.show() 
    df_pvalues = pd.DataFrame(p_values_matrix)
    df_pvalues.to_csv(path2+'Pvalues/pvalues_p53_divisions'+str(labels[i])+'.csv', index=False)
    
    df_list=pd.DataFrame().from_dict(amp_p21, orient='index')
    df_list=df_list.T
    upper_limit = df_list[0].quantile(0.75) + 1.5 * (df_list[0].quantile(0.75) - df_list[0].quantile(0.25))
    high = int(df_list[0][df_list[0] <= upper_limit].max())
    medians = df_list.median()
    corr = np.corrcoef(np.arange(0, max_div, step=1), medians)[0,1]
    se = np.sqrt((1 - corr**2) / (len(medians) - 2))
    # print(corr, se)
    p_values_matrix = np.zeros((max_div, max_div))   
    fig = plt.figure(figsize = (12,10) )
    df_list.boxplot(showfliers = False)
    for jj in range(max_div):
        for ii in range(max_div):
            if ii > jj:
                t_stat, p_value = ttest_ind(df_list[jj].dropna(), df_list[ii].dropna(), equal_var=False)
                print("P-value of "+str(jj)+" to "+str(ii)+":", p_value)
                p_values_matrix[jj,ii] = p_value
                if jj+1 == ii:
                    t_stat, p_value = ttest_ind(df_list[jj].dropna(), df_list[ii].dropna(), equal_var=False)
                    plt.text(1.5+1*jj, high, f" {p_value:.2e}", fontsize=12, ha='center', va='center')
    plt.text(max_div-2, high*0.75, r'Corr coeff: {:.2f} $\pm$ {:.2f}'.format(corr,se))
    plt.ylabel('p21 amplitude (a.u.)')
    plt.xlabel('Number divisions')
    # plt.savefig(path2+'Boxplot_ampl_p21_divisions_'+str(labels[i])+'.svg')
    plt.show() 
    df_pvalues = pd.DataFrame(p_values_matrix)
    df_pvalues.to_csv(path2+'Pvalues/pvalues_p21_divisions'+str(labels[i])+'.csv', index=False)


    
    
    