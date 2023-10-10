#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 16:48:23 2023

@author: nicagutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
from pyboat import ensemble_measures as em
from pyboat import plotting as pl 
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from scipy.stats import ttest_ind

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'

path='Dataset/'
path2='Output/'

#Untreated U2OS experiment 50hours
dishes=['2','3','4','5'] 

imt = {str(num): [] for num in range(2, 4)}
for i in dishes:
    gfp_file='gfp_dish'+str(i)+'_untreated'
    divisions_file='divisions_dish'+str(i)+'_untreated'
    data_gfp=pd.read_excel(path+gfp_file+r'.xlsx',header=None)
    data_gfp=data_gfp.T
    divisions=pd.read_excel(path+divisions_file+r'.xlsx',header=None) 
    divisions=divisions.T
    
    for col in data_gfp:
        if data_gfp[col][0]==-1:
            data_gfp=data_gfp.drop(col, axis=1)
            divisions=divisions.drop(col, axis=1)
            
    for col in divisions:
        if len((np.where(divisions[col]==1)[0]))!=0:
            division_times=((np.where(divisions[col]==1)[0]))
            if len(division_times)>1:
                for ii in range(len(division_times)-1):
                    if (division_times[ii+1]-division_times[ii])<18:
                        if col in divisions:
                            divisions=divisions.drop(col, axis=1)                   

    for col in divisions:
        division_times=((np.where(divisions[col]==1)[0]))
        imt_single=[]
        if 2 <= len(division_times) <= 5:
            for ii in range(len(division_times)-1):
                imt_single.append((division_times[ii+1]-division_times[ii]))
            imt[str(len(division_times))].append(np.mean(imt_single))
            

df_imt=pd.DataFrame.from_dict(imt, orient='index')
df_imt=df_imt.transpose()
df_imt.columns=[str(num)+' divisions' for num in range(2, 4)]
t_stat, p_value = ttest_ind(df_imt[df_imt.columns[1]].dropna(), df_imt[df_imt.columns[0]].dropna(), equal_var=False)
print("P-value of "+str(df_imt.columns[1])+" to "+str(df_imt.columns[0])+":", p_value)

fig=plt.figure(figsize=(10,10))
df_imt.boxplot(showfliers=False)
plt.ylabel('IMT (h)')
# plt.savefig(path2+'IMT_boxplot_divisions_U2OS_untreated_short.svg')
plt.show()

# fig=plt.figure(figsize=(10,10))
# # df_imt.violin(showfliers=False)
# sns.violinplot(data=df_imt)
# plt.ylabel('IMT (h)')
# # plt.savefig(path2+'IMT_violinplot_numdivisions_U2OS_untreated.svg')
# plt.show()







