#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:30:25 2023

@author: nicagutu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'

path='Dataset/'
path2='Output/'
file='DataSetTraces_09052018'
file1='DishPosCellId_09052018'

data=pd.read_csv(path+file+r'.csv')
IDs_data=pd.read_csv(path+file1+r'.csv')

cond_list = {'untreated': [], 'cisplatin': []}
cond_list['cisplatin'] = IDs_data[(IDs_data['Dish'] >= 10) & (IDs_data['Dish'] <= 15)]['Cell_id'].tolist()
cond_list['untreated'] = IDs_data[(IDs_data['Dish'] >= 22) & (IDs_data['Dish'] <= 24)]['Cell_id'].tolist()
        
divisions=pd.DataFrame()
fig=plt.figure(figsize=(12,10))
for ii in cond_list:
    IDs=cond_list[ii]
    low_ID=[]
    high_ID=[]
    for col in IDs:
        divs=data['DivisionEvents_CellId_'+str(col)][0:96]
        deaths=data['DeathEvents_CellId_'+str(col)]
        time=deaths.index.values
        death=(np.where(deaths==1)[0])
        if sum(divs)<=1:
            low_ID.append(col)
        elif sum(divs)>=2:
            high_ID.append(col)
    
    for j in low_ID:
        divisions[j]=np.array(data['DivisionEvents_CellId_'+str(j)])[0:96]
    for j in high_ID:
        divisions[j]=np.array(data['DivisionEvents_CellId_'+str(j)])[0:96]

    division_profile0 = np.array([[data['DivisionEvents_CellId_'+str(j)]] for j in low_ID])                    
    distr0 = np.sum(division_profile0[:len(low_ID), :], axis=0)
       
    division_profile1 = np.array([[data['DivisionEvents_CellId_'+str(j)]] for j in high_ID])                    
    distr1 = np.sum(division_profile1[:len(low_ID), :], axis=0)
    
    plt.plot(time*0.5,np.cumsum(distr0)/max(np.cumsum(distr1)),label='Low prolif '+str(ii))
    plt.plot(time*0.5,np.cumsum(distr1)/max(np.cumsum(distr1)),label='High prolif '+str(ii))
plt.xlabel('Time[h]')
plt.ylabel('Cumulative distribution of division events')
plt.legend(loc='best')
# plt.savefig(path2+'Sensitivity_cum_distr_division.svg')
plt.show()

#%%Division profile
for col in divisions:
    div_events=np.where(divisions[col]==1)[0]
    if sum(data['DeathEvents_CellId_'+str(col)])==1:
        divisions=divisions.drop(col,axis=1)
    if len(div_events)>1:
        for ii in range(len(div_events)-1):
            if (div_events[ii+1]-div_events[ii])<40:
                if col in divisions:
                    divisions=divisions.drop(col, axis=1)                

time_points = len(divisions.index.values)                                                         
properties = {'num_divisions':[],'time_1st':[]}
IDs = divisions.columns

for col in IDs:
    properties['num_divisions'].append(sum(divisions[col]))
    count=0
    for jj in range(len(divisions[col])):
        if sum(divisions[col]) == 0 and count <1:
            count+=1
            properties['time_1st'].append(0)
        elif sum(divisions[col]) != 0:
            if divisions[col][jj] == 1 and count <1:
                properties['time_1st'].append(divisions.index.values[jj])
                count+=1
            
properties['num_divisions'],properties['time_1st'],IDs=zip(*sorted(zip(properties['num_divisions'],properties['time_1st'],IDs)))

divisions = divisions.reindex(columns=IDs)
division_profile = (divisions.to_numpy())
division_profile = division_profile.T

for xx in range(len(IDs)):
    divisions = []
    for yy in range(time_points):
        if division_profile[xx,yy] == 1:
            divisions.append(yy)
    if len(divisions) > 0:
        for n in range(len(divisions)-1):
            division_profile[xx,divisions[n]+1:divisions[n+1]] = division_profile[xx,divisions[n]+1:divisions[n+1]]+(n+1)   
        division_profile[xx,divisions[-1]+1:time_points] = division_profile[xx,divisions[-1]+1:time_points]+len(divisions)
    if len(divisions) > 1:    
        division_profile[xx,divisions[-1]] = division_profile[xx,divisions[-1]]+1
 
plt.figure(figsize=(8, 14))
color_map=plt.imshow(division_profile)
color_map.set_cmap("plasma")
plt.xlabel('Time(h)')
plt.ylabel('Cell number')
ax=plt.gca()
divider=make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
cb=plt.colorbar(cax=cax)
labels_list=np.arange(0,sum(divisions),1)
loc=labels_list+0
cb.set_ticks(loc)
cb.set_ticklabels(labels_list)
cb.set_label('No. divisions',rotation=270, labelpad=25)
# plt.savefig(path2+'Cisplatin_sensitivity_division_profile.svg')
plt.show()

