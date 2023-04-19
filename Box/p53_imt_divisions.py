#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 13:44:31 2023

@author: nicagutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
from pyboat import ensemble_measures as em
from pyboat import plotting as pl 
import seaborn as sns
import random
from scipy.stats import kurtosis
import scipy.stats as stats
from scipy.stats import ttest_ind

plt.rcParams.update({'font.size': 22})    
plt.rcParams['svg.fonttype'] = 'none'

path='Dataset/'
path2='Output/'
file_list=['0G','2G','4G','10G_MD','10G_NMD']
labels=['0Gy','2Gy','4Gy','10Gy']
colors=['tab:green','tab:brown']

# fig = plt.figure(figsize = (12,10))
for i in range(1):
    file = file_list[i] 
    print(file)
    divisions = pd.read_csv(path+'division_matrix_'+file_list[i]+r'.csv',header=None)
    divisions = divisions.T
    
    num_divisions = []
    imt = {str(num): [] for num in range(2, 6)}
    imt_all = []
    for col in divisions:
        div_events = (divisions[col])
        division_times = ((np.where(div_events==1)[0]))
        
        num_divisions.append(len(division_times))
        imt_single = []
        if 2 <= len(division_times) <= 5:
            for ii in range(len(division_times)-1):
                imt_single.append((division_times[ii+1]-division_times[ii])*0.5)
            imt[str(len(division_times))].append(np.mean(imt_single))
            # imt_all.append(np.mean(imt_single))
            
    df_imt = pd.DataFrame.from_dict(imt, orient='index')
    df_imt = df_imt.transpose()
    df_imt.columns = [str(num)+' divisions' for num in range(2, 6)]
    
    print('Mean value ', np.mean(df_imt.mean()))
    print('Median value ', np.mean(df_imt.median()))
    print('Coefficient of variation of means ', np.std(df_imt.mean())/np.mean(df_imt.mean()))
    
    for jj in range(4):
        coljj = df_imt.columns[jj]
        print('n ',len(df_imt[coljj].dropna()))
    #     for ii in range(4):
    #         colii = df_imt.columns[ii]
    #         if ii > jj:
    #             if jj+1 == ii:
    #                 t_stat, p_value = ttest_ind(df_imt[coljj].dropna(), df_imt[colii].dropna(), equal_var=False)
    #                 print("P-value of "+str(coljj)+" to "+str(colii)+":", p_value)

    for jj in range(1,4):
        coljj = df_imt.columns[jj]
        t_stat, p_value = ttest_ind(df_imt[coljj].dropna(), df_imt[df_imt.columns[0]].dropna(), equal_var=False)
        print("P-value of "+str(coljj)+" to "+str(df_imt.columns[0])+":", p_value)

    # fig = plt.figure(figsize=(10,10))
    # df_imt.boxplot(showfliers=False)
    # plt.ylabel('IMT (h)')
    # # plt.savefig(path2+'IMT_boxplot_numdivisions_p53_'+str(labels[i])+'.svg')
    # plt.show()
    
    # print(len(num_divisions))
    # print('Mean ',np.mean(num_divisions))
    # print('SD ',np.std(num_divisions))
    # print('Kurtosis ', kurtosis(num_divisions))
    
    # counts = np.bincount(num_divisions) 
    # probs = counts/len(num_divisions)
    # entropy = stats.entropy(probs)
    # print('Entropy ', entropy)
    
    
    # fig = plt.figure(figsize = (12,10))
    # sns.histplot(data=num_divisions, bins=[0, 1, 2, 3, 4], discrete=True, stat='density', label='RPE untreated')
    # plt.xticks([0,1,2,3,4,5])
    # plt.xlabel('Total number of divisions')
    # plt.legend(loc='best')
    # # plt.savefig(path2+'Heterogeneity_prolif_RPE_'+str(labels[i])+'.svg')
    # plt.show()

#     sns.kdeplot(data=imt_all, label='RPE '+str(labels[i]), color=colors[i])
# plt.xlabel('IMT [hours]')
# # plt.xlim([0,60])
# plt.legend(loc='best')
# plt.savefig(path2+'IMT_all_RPE_'+str(labels[i])+'.svg')
# plt.show()  
    