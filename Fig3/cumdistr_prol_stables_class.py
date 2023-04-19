#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 13:47:39 2023

@author: nicagutu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import random

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'      

path = 'Dataset/'
path2 = 'Output/'

files = ['2G', '4G', '10G_MD']
labels = ['2Gy', '4Gy', '10Gy']
colors = ['tab:blue', 'tab:green', 'tab:purple']
colors2 = ['blue', 'green', 'purple']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label = 'hours')

# #%%All cells
# fig = plt.figure(figsize = (12,10))
# for i in range(3):
#     data1 = pd.read_csv(path+'Constperiod_'+files[i]+r'.csv', header = 0, index_col = 0) 
#     data2 = pd.read_csv(path+'Incrperiod_'+files[i]+r'.csv',header = 0,index_col = 0) 
    
#     division_matrix = pd.read_csv(path+'division_matrix_'+files[i]+r'.csv', header = None)
#     division_matrix = division_matrix.T
#     time = division_matrix.index.values*0.5
    
#     distr = np.zeros((2, len(division_matrix)))
#     for col, ii in zip([data1, data2], [0, 1]):
#         division_profile = pd.DataFrame()
#         for c in col:
#             division_profile[c] = division_matrix[int(c)]
#         distr[ii] = np.sum(division_profile.to_numpy().T[:len(col.columns), :], axis = 0)
        
#     maximum = max(np.cumsum(distr[0]).max(), np.cumsum(distr[1]).max())    
    
#     plt.plot(time, np.cumsum(distr[0])/maximum,'--', label='Const '+str(labels[i]), color = colors[i])
#     plt.plot(time, np.cumsum(distr[1])/maximum, label='Increase '+str(labels[i]), color = colors[i])

# plt.xlabel('Time(h)')
# plt.ylabel('Cumulative distribution of division events')
# plt.legend(loc='best')
# # plt.savefig(path2+'Cumul_distr_incr_const_all.svg')
# plt.show()


#%%Bootstrapping to avoid sample size effects

fig = plt.figure(figsize = (12,10))
for i in range(3):
    data1 = pd.read_csv(path+'Constperiod_'+files[i]+r'.csv', header = 0, index_col = 0) 
    data2 = pd.read_csv(path+'Incrperiod_'+files[i]+r'.csv',header = 0,index_col = 0) 
    
    division_matrix = pd.read_csv(path+'division_matrix_'+files[i]+r'.csv', header = None)
    division_matrix = division_matrix.T
    time = division_matrix.index.values*0.5
    sample_size=min(len(data1.columns),len(data2.columns))
                
    median_distr1=pd.DataFrame()
    median_distr2=pd.DataFrame()
    for a in range(100):    
        num_Ids1=random.sample(list(data1.columns),sample_size)
        num_Ids2=random.sample(list(data2.columns),sample_size)
        
        division_profile1 = np.array([division_matrix[int(j)] for j in num_Ids1])                    
        distr1 = np.sum(division_profile1[:len(data1.columns), :], axis=0)
            
        division_profile2 = np.array([division_matrix[int(j)] for j in num_Ids2])                    
        distr2= np.sum(division_profile2[:len(data2.columns), :], axis=0)

        maximum=max(max(np.cumsum(distr1)),max(np.cumsum(distr2)))
    
        median_distr1[a]=distr1/(maximum)
        median_distr2[a]=distr2/(maximum)
        
    average_distr1=median_distr1.mean(axis=1)
    std1=median_distr1.std(axis=1)
    average_distr2=median_distr2.mean(axis=1)
    std2=median_distr2.std(axis=1)
        
    df_const=pd.DataFrame(data=np.cumsum(average_distr1),columns=['Cum distr'])
    df_const.to_csv(path+'Constant_cum_distr_'+files[i]+'.csv')
    df_incr=pd.DataFrame(data=np.cumsum(average_distr2),columns=['Cum distr'])
    df_incr.to_csv(path+'Increase_cum_distr_'+files[i]+'.csv')

    plt.errorbar(time,np.cumsum(average_distr1),yerr=std1,label='Constant '+str(labels[i]),color=colors[i])
    plt.errorbar(time,np.cumsum(average_distr2),yerr=std2,label='Increasing '+str(labels[i]),color=colors2[i])
    
    cum_distr1 = np.cumsum(average_distr1)
    cum_distr2 = np.cumsum(average_distr2)
    slope1, intercept1 = np.polyfit(time[120:240], cum_distr1[120:240], 1)
    slope2, intercept2 = np.polyfit(time[120:240], cum_distr2[120:240], 1)
    print('slope switchers and stables of', files[i], slope2, slope1)
    # print('relative slope between switchers and stables of', files[i], slope2/slope1)
    
plt.xlabel('Time(h)')
plt.ylabel('Cumulative distribution of division events')
plt.legend(loc='best')
# plt.savefig(path2+'Cumul_distr_all_divisions_bootstr_STD.svg')
plt.show()



