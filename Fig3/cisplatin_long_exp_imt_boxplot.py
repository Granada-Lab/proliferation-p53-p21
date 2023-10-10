#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:19:22 2023

@author: nicagutu
"""

import pandas as pd
import kaplanmeier as km
from lifelines import KaplanMeierFitter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kurtosis
import scipy.stats as stats
from scipy.stats import ttest_ind

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})    

path='Dataset/'
file='DataSetTraces_09052018'
file1='DishPosCellId_09052018'
path2='Output/'

data=pd.read_csv(path+file+r'.csv')
IDs_data=pd.read_csv(path+file1+r'.csv')

IDs_untreated = [cell_id for cell_id, dish in zip(IDs_data['Cell_id'], IDs_data['Dish']) if 22 <= dish <= 24]

num_divisions = []
imt = {str(num): [] for num in range(2, 6)}
# iterate over the cell ids
for col in IDs_untreated:
    divisions = data['DivisionEvents_CellId_'+str(col)]
    division_times = np.where(divisions==1)[0]
    
    num_divisions.append(len(division_times))
    if 2 <= len(division_times) <= 5:
        imt_single = []
        # calculate the intermitotic time
        for ii in range(len(division_times)-1):
            imt_single.append((division_times[ii+1]-division_times[ii])*0.5)
        # append the mean intermitotic time to the corresponding list
        imt[str(len(division_times))].append(np.mean(imt_single))
                    
df_imt=pd.DataFrame.from_dict(imt, orient='index')
df_imt=df_imt.transpose()
df_imt.columns=[str(num)+' divisions' for num in range(2, 6)]

print('Mean value ', np.mean(df_imt.mean()))
print('Median value ', np.mean(df_imt.median()))
print('Coefficient of variation of means ', np.std(df_imt.mean())/np.mean(df_imt.mean()))
for jj in range(4):
    coljj = df_imt.columns[jj]
    print('n ',len(df_imt[coljj].dropna()))
    for ii in range(4):
        colii = df_imt.columns[ii]
        if ii > jj:
            if jj+1 == ii:
                t_stat, p_value = ttest_ind(df_imt[coljj].dropna(), df_imt[colii].dropna(), equal_var=False)
                print("P-value of "+str(coljj)+" to "+str(colii)+":", p_value)

for jj in range(1,4):
    coljj = df_imt.columns[jj]
    t_stat, p_value = ttest_ind(df_imt[coljj].dropna(), df_imt[df_imt.columns[0]].dropna(), equal_var=False)
    print("P-value of "+str(coljj)+" to "+str(df_imt.columns[0])+":", p_value)

fig=plt.figure(figsize=(10,10))
df_imt.boxplot(showfliers=False)
plt.ylabel('IMT [hours]')
# plt.savefig(path2+'IMT_boxplot_divisions_U2OS_untreated_long.svg')
plt.show()

fig = plt.figure(figsize = (12,10))
sns.histplot(data=num_divisions, bins=[0, 1, 2, 3, 4], discrete=True, stat='density', label='U2OS untreated')
plt.xticks([0,1,2,3,4,5])
plt.xlabel('Total number of divisions')
plt.legend(loc='best')
# plt.savefig(path2+'Heterogeneity_prolif_U2OS_untreated_long.svg')
plt.show()

print(len(num_divisions))
print('Mean ',np.mean(num_divisions))
print('SD ',np.std(num_divisions))
print('Kurtosis ', kurtosis(num_divisions))

counts = np.bincount(num_divisions) 
probs = counts/len(num_divisions)
entropy = stats.entropy(probs)
print('Entropy ', entropy)



        
        