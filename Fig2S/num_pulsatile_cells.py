#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 13:43:23 2022

@author: nicagutu
"""

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks
from pyboat import WAnalyzer
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rcParams['svg.fonttype'] = 'none'

path = 'Dataset/'
path2 = 'Output/'

file_list = ['0G', '2G', '4G', '10G_MD']

colors_1 = ['midnightblue','tab:blue','royalblue','slateblue','mediumpurple','indigo','darkorchid']
labels = ['0G','2G','4G','10G']

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

thresh = 10
pulsatile_cells = []
for i in range(4):
    file = file_list[i]
      
    data = pd.read_csv(path+'p53_'+file+r'.csv',header=None)
    time = data.index.values
    
    count = 0
    for column in data:
        signal = data[column] 
        # signal = wAn.sinc_detrend(signal, T_c=50)
        peaks = find_peaks(signal,height=50,distance=7,prominence=50)
        if len(peaks[0]) >= thresh:
            count += 1
            # plt.plot(time, signal)
            # plt.show()
    #     else:
    #         plt.plot(time, signal)
    #         plt.show()
    # print(count,len(data.columns))
    pulsatile_cells.append(count/len(data.columns)*100) 
 
fig = plt.figure(figsize=(10,10))
barlist = plt.bar(labels, pulsatile_cells)
for ii in range(4):
    barlist[ii].set_color(colors_1[ii])
plt.ylabel('% cells with > '+str(thresh)+' pulses/120 hours')
# plt.savefig(path2+'Number_osc_cells_condition.svg')
plt.show()
print(pulsatile_cells)