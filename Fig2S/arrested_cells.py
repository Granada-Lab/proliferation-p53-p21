#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:01:05 2023

@author: nicagutu
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyboat import WAnalyzer
from pyboat import ensemble_measures as em
from pyboat import plotting as pl 
import seaborn as sns

plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'

path='Dataset/'
path2='Output/'

files=['0G', '2G', '4G', '10G_MD']
labels=['0Gy','2Gy','4Gy','10Gy']

df_cum_div=pd.DataFrame(columns=labels)  
fraction_cells=[]
for i in range(len(files)):
    file='division_matrix_'+str(files[i])

    data=pd.read_csv(path+file+r'.csv')
    data=data.T
    time=np.arange(0,len(data.index.values),step=1)
    data.set_index(time,inplace=True)
    
    num_Ids=data.columns
    time_points=len(data.index.values)
    division_profile=np.zeros((len(num_Ids),time_points))
    
    for j in range(len(num_Ids)):
        vect=[]
        count=0
        for k in range(time_points):
            #Counting only first divison
            if data.iloc[k,j]==0 and count<1:
                vect.append(data.iloc[k,j])
            elif data.iloc[k,j]==1 and count<1 and k<=96:
                count+=1
                vect.append(data.iloc[k,j])
            else:#if count>=1:
                vect.append(0)                
        division_profile[j,:]=vect
                        
    distr = np.sum(division_profile[:len(data.columns), :], axis=0)

    df_cum_div[labels[i]]=1-np.cumsum(distr)/len(data.columns)
    fraction_cells.append(min(1-np.cumsum(distr)/len(data.columns)))

print(df_cum_div)
fig=plt.figure(figsize=(12,10))
last_60 = df_cum_div.tail(200)
sns.barplot(data=last_60)
plt.ylabel('Fraction of long-term arrested cells')
plt.savefig(path2+'Long_term_arrested_fraction.svg')
plt.show()

fig=plt.figure(figsize=(12,10))
for ii in range(len(files)):
    plt.plot(time*0.5,df_cum_div[labels[ii]],label=str(labels[ii]))
plt.xlabel('Time(h)')
plt.ylabel('Long-term arrested cells')#'division events')
plt.ylim([0,1])
plt.legend(loc='best')
# plt.savefig(path2+'Long_term_arrested_cumul_distr.svg')
plt.show()


