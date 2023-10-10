#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 11:54:05 2023

@author: nicagutu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import seaborn as sns
from scipy.stats import ttest_ind, pearsonr
import scipy.stats as stats

plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'

path='Dataset/'
path2='Output/'

file_list=['0G', '2G', '4G', '10G_MD']
labels=['0Gy','2Gy','4Gy','10Gy']

dt = 0.5 # the sampling interval, 0.5hours
lowT = 3
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

for i in range(4):  
    file = file_list[i]    
    data = pd.read_csv(path+'p53_'+file+r'.csv',header=None) 
    time = data.index.values*0.5
    print(file)
    
    periods_all = []
    amplitudes_all = []
    for column in data:
        signal = data[column]
        detrended = wAn.sinc_detrend(signal, T_c=50)
        wAn.compute_spectrum(detrended,do_plot=False)
        rd = wAn.get_maxRidge(power_thresh=0,smoothing_wsize=5)
        
        for ii in range(len(rd['periods'])):
            if rd['periods'][ii] < 10:
                periods_all.append(rd['periods'][ii])
                amplitudes_all.append(rd['amplitude'][ii])

    # Calculate the correlation coefficient and error
    corr_coef, p_value = pearsonr(periods_all, amplitudes_all)
    corr_error = np.sqrt((1 - corr_coef**2) / (len(periods_all) - 2))
    print("Correlation coefficient:", corr_coef)
    print("Correlation error:", corr_error)

    #Plot joint distribution plot
    fig = plt.figure(figsize=(12,12))
    kdeplot = sns.jointplot(x=periods_all, y=amplitudes_all, fill=True, kind='kde', cbar=True, xlim=[3,11], ylim=[0,500])
    plt.text(-0.07, 450, 'Corr.='+str(round(corr_coef,3))+'$\pm$'+str(round(corr_error,3)), fontsize=20, ha='center', va='center')
    plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
    pos_joint_ax = kdeplot.ax_joint.get_position()
    pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
    kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
    kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
    kdeplot.ax_joint.set_xlabel('Period(h)')
    kdeplot.ax_joint.set_ylabel('Amplitude')
    plt.suptitle(str(labels[i]))
    # plt.savefig(path2+'Joint_distr_ampl_period_'+str(labels[i])+'.svg')
    plt.show()
    


