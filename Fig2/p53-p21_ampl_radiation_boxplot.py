#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:26:32 2023

@author: nicagutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
from scipy.stats import ttest_ind

plt.rcParams.update({'font.size': 26})    
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2 = 'Output/'

file_list = ['0G','2G','4G','10G_MD','10G_NMD']
labels = ['0Gy','2Gy','4Gy','10Gy']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

dict_ampl_p53 = {}
dict_ampl_p21 = {}
for i in range(4):
    file = file_list[i]
    ampl_p53 = []
    ampl_p21 = []
    
    datap53 = pd.read_csv(path+'p53_'+file+r'.csv', header = None) 
    datap21 =pd.read_csv(path+'p21_'+file+r'.csv', header = 0, index_col = 0) 
    datap53.columns = datap21.columns
            
    for column in datap53:
        signalp53 = datap53[column]
        detrended_signalp53 = wAn.sinc_detrend(signalp53, T_c = 50)
        wAn.compute_spectrum(detrended_signalp53, do_plot = False) 
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize = 4) 
        rdp53 = wAn.ridge_data

        signalp21 = datap21[column]
        wAn.compute_spectrum(signalp21, do_plot = False) 
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize = 4) 
        rdp21 = wAn.ridge_data
        
        ampl_p53.append(np.median(rdp53['amplitude']))
        ampl_p21.append(np.median(rdp21['amplitude']))
    
    dict_ampl_p53[i] = ampl_p53
    dict_ampl_p21[i] = ampl_p21

df_ampl_p53 = pd.DataFrame.from_dict(dict_ampl_p53, orient='index')
df_ampl_p53 = df_ampl_p53.transpose()
df_ampl_p53.columns = labels
upper_limit = df_ampl_p53[labels[3]].quantile(0.75) + 1.5 * (df_ampl_p53[labels[3]].quantile(0.75) - df_ampl_p53[labels[3]].quantile(0.25))
high = df_ampl_p53[labels[3]][df_ampl_p53[labels[3]] <= upper_limit].max()    
medians = df_ampl_p53.median()
corr = np.corrcoef(np.arange(0, 4, step=1), medians)[0,1]
se = np.sqrt((1 - corr**2) / (len(medians) - 2))

p_values_matrix = np.zeros((4, 4))   
fig = plt.figure(figsize = (12,10))
df_ampl_p53.boxplot(showfliers = False)
# Compute the t-test p-value
for jj in range(4):
    for ii in range(4):
        if ii > jj:
            t_stat, p_value = ttest_ind(df_ampl_p53[labels[jj]].dropna(), df_ampl_p53[labels[ii]].dropna(), equal_var=False)
            print("P-value of "+str(jj)+" to "+str(ii)+":", p_value)
            p_values_matrix[jj,ii] = p_value
            if jj+1 == ii:
                t_stat, p_value = ttest_ind(df_ampl_p53[labels[jj]].dropna(), df_ampl_p53[labels[ii]].dropna(), equal_var=False)
                plt.text(1.5+1*jj, high, f" {p_value:.2e}", fontsize=12, ha='center', va='center')
plt.text(2, high*0.75, r'Corr coeff: {:.2f} $\pm$ {:.2f}'.format(corr,se))
plt.ylabel('p53 amplitude [a.u.]')
plt.xlabel('Radiation dose')
# plt.savefig(path2+'p53_boxplot_ampl_doses.svg')
plt.show()
df_pvalues = pd.DataFrame(p_values_matrix)
df_pvalues.to_csv(path2+'Pvalues/pvalues_p53_dose'+str(labels[i])+'.csv', index=False)


df_ampl_p21 = pd.DataFrame.from_dict(dict_ampl_p21, orient = 'index')
df_ampl_p21 = df_ampl_p21.transpose()
df_ampl_p21.columns = labels
upper_limit = df_ampl_p21[labels[3]].quantile(0.75) + 1.5 * (df_ampl_p21[labels[3]].quantile(0.75) - df_ampl_p21[labels[3]].quantile(0.25))
high = df_ampl_p21[labels[3]][df_ampl_p21[labels[3]] <= upper_limit].max()    
medians = df_ampl_p21.median()
corr = np.corrcoef(np.arange(0, 4, step=1), medians)[0,1]
se = np.sqrt((1 - corr**2) / (len(medians) - 2))
print(corr, se)

p_values_matrix = np.zeros((4, 4))   
fig = plt.figure(figsize = (12,10))
df_ampl_p21.boxplot(showfliers = False)
# Compute the t-test p-value
for jj in range(4):
    for ii in range(4):
        if ii > jj:
            t_stat, p_value = ttest_ind(df_ampl_p21[labels[jj]].dropna(), df_ampl_p21[labels[ii]].dropna(), equal_var=False)
            print("P-value of "+str(jj)+" to "+str(ii)+":", p_value)
            p_values_matrix[jj,ii] = p_value
            if jj+1 == ii:
                t_stat, p_value = ttest_ind(df_ampl_p21[labels[jj]].dropna(), df_ampl_p21[labels[ii]].dropna(), equal_var=False)
                plt.text(1.5+1*jj, high, f" {p_value:.2e}", fontsize=12, ha='center', va='center')
plt.text(2, high*0.75, r'Corr coeff: {:.2f} $\pm$ {:.2f}'.format(corr,se))
plt.xlabel('Radiation dose')
plt.ylabel('p21 amplitude [a.u.]')
# plt.savefig(path2+'p21_boxplot_ampl_doses.svg')
plt.show()
df_pvalues = pd.DataFrame(p_values_matrix)
df_pvalues.to_csv(path2+'Pvalues/pvalues_p21_dose'+str(labels[i])+'.csv', index=False)




