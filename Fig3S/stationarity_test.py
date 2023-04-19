#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:44:26 2023

@author: nicagutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

plt.rcParams.update({'font.size': 26})    
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2 = 'Output/'
colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple']

file_list = ['0G', '2G', '4G', '10G_MD', '10G_NMD']
labels = ['0Gy', '2Gy', '4Gy', '10Gy']

fig=plt.figure(figsize=(10,8))
pvalues_all = {'0Gy':[], '2Gy':[], '4Gy':[], '10Gy':[]}
fractions = []
for i in range(4):
    file = file_list[i]    
    datap53 = pd.read_csv(path+'p53_'+file+r'.csv', header = None) 
    datap21 =pd.read_csv(path+'p21_'+file+r'.csv', header = 0, index_col = 0) 
    datap53.columns = datap21.columns
            
    p_values=[]
    count=0
    for column in datap53:
        signalp53 = datap53[column]
        result = sm.tsa.stattools.adfuller(signalp53)
        p_values.append(result[1])
        if result[1]>0.05:
            count+=1
    fractions.append(100*count/len(datap53.columns))
    pvalues_all[labels[i]] = p_values
    sns.histplot(p_values, kde=True, bins=50, stat='count', label=labels[i], color=colors[i])
    
plt.xlabel('p-values')
plt.ylabel('Count')
plt.legend(loc='best')
# plt.savefig(path2+'Stationarity_test_pvalues.svg')
plt.show()

fig=plt.figure(figsize=(10,8))
plt.bar(labels, fractions)
plt.axhline(50, linestyle='--', color='gray')
plt.xlabel('Radiation dose')
plt.ylabel('% of non-stationary cells')
plt.ylim([0,100])
# plt.savefig(path2+'Stationarity_test_fraction.svg')
plt.show()



