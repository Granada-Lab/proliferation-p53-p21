#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:04:14 2023

@author: nicagutu
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import warnings

plt.rcParams.update({'font.size': 22})    
warnings.filterwarnings("ignore")
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2 = 'Output/'

files = ['0G', '2G', '4G', '10G_MD']
labels=['0Gy', '2Gy', '4Gy','10G_MD']

colors = ['black','tab:blue','tab:green','tab:purple']
colors2 = ['grey','blue','green','purple']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

fig = plt.figure(figsize=(12, 10))

#P2D plot with two y-axis
ax1 = fig.add_subplot(111)
ax2= ax1.twinx()

# #3D plot
# ax = fig.add_subplot(111, projection='3d')
# y = {'1': np.zeros(240), '2': np.ones(240), '3': np.full(240, 2)}
for i in range(1,4):
    period_file1 = pd.read_csv(path+'Constperiod_'+files[i]+r'.csv',index_col=0,header=0)
    period_file2 = pd.read_csv(path+'Incrperiod_'+files[i]+r'.csv',index_col=0,header=0)
    time = period_file1.index.values*0.5
    diff_per = abs(period_file1.median(axis=1)-period_file2.median(axis=1))
    trend_period = wAn.sinc_smooth(diff_per, T_c=10) #looks for the trend
    
    cumdistr_file1 = pd.read_csv(path+'Constant_cum_distr_'+files[i]+r'.csv',index_col=0,header=0)
    cumdistr_file2 = pd.read_csv(path+'Increase_cum_distr_'+files[i]+r'.csv',index_col=0,header=0)
    diff_cumdistr = abs(cumdistr_file1['Cum distr']-cumdistr_file2['Cum distr'])
    cumdistr_trend = wAn.sinc_smooth(diff_cumdistr, T_c=10) #looks for the trend
    
    # slope1 = np.mean(np.gradient(cumdistr_trend[0:120]))
    # slope2 = np.mean(np.gradient(cumdistr_trend[120:240]))
    # print(slope2/slope1)
    
    # slope1, intercept1 = np.polyfit(time[20:100], cumdistr_trend[20:100], 1)
    # slope2, intercept2 = np.polyfit(time[120:240], cumdistr_trend[120:240], 1)
    # print('fold change of slope of', files[i], slope2, slope1)
    
    ax1.plot(time, trend_period,color=colors[i],label='Period '+labels[i])
    ax2.plot(time, cumdistr_trend,'--',color=colors2[i],label='Cumulative '+labels[i])
    
    # ax.plot(time,y[str(i)],trend_period,color=colors[i],label='Period '+labels[i])
    # ax.plot(time,y[str(i)],cumdistr_trend,'--',color=colors2[i],label='Cumulative '+labels[i])


#Set the labels and parameters for the 2D plot with 2 y-axixs
ax1.set_xlabel('Time [h]')
ax1.set_ylabel('Period difference [h]')
ax2.set_ylabel('Cumulative distribution difference [h]')
ax1.legend(bbox_to_anchor=(0.35, 0.8))#(loc='center left')
ax2.legend(loc='upper left')
# plt.savefig(path2+'Difference_periods_cumdistr.svg')
plt.show()

# # Set the labels for the x, y and z axis of the 3D plot
# ax.set_xlabel('Time [hours]',labelpad=15)
# ax.set_ylabel('Radiation dose',labelpad=15)
# ax.set_zlabel('Difference',labelpad=15)
# ax.set_yticklabels([None,'2Gy',None,'4Gy',None,'10Gy'])
# ax.view_init(elev=30, azim=-55)
# plt.savefig(path2+'Period_cumdistr_diff_3Dplot.svg')
# plt.show()



