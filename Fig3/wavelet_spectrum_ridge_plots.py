#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 12:09:17 2023

@author: nicagutu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyboat import WAnalyzer

plt.rcParams.update({'font.size': 22})
plt.rcParams['svg.fonttype'] = 'none'      

path='Dataset/'
path2='Output/'

files=['0G', '2G', '4G', '10G_MD']
labels=['0Gy','2Gy','4Gy','10Gy']
colors=['tab:blue','tab:orange','tab:green','tab:purple']
divisions=[0,1,2,3,4,5]

#Setting the parameters necessary for the Wavelet spectrum analysis
dt = 0.5 # the sampling interval, 0.5hours
lowT = 2
highT = 10
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label = 'hours')

for i in range(1,4):#len(files)):
    data=pd.read_csv(path+'p53_'+files[i]+r'.csv', header = None, index_col = None) 
    time=data.index.values*0.5
    
    for col in data:
        signal=data[col]
        detrended_signal=wAn.sinc_detrend(signal, T_c = 50)
        norm_amp = wAn.normalize_amplitude(detrended_signal, window_size=50)
        wAn.compute_spectrum(norm_amp, do_plot = False) 
        wAn.get_maxRidge(power_thresh = 0, smoothing_wsize = 4) 
        rd=wAn.ridge_data 
        
        if np.std(rd['periods'])>0 and np.std(rd['periods'])<2 and int(col)<10:
            wAn.compute_spectrum(detrended_signal, do_plot=True) 
            wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
            # plt.savefig(path2+str(files[i])+'_p53_spectrum_'+str(count)+'.svg')
            plt.show()
            
            rd=wAn.ridge_data 
            wAn.draw_Ridge()            
            wAn.plot_readout(draw_coi=True)
            # plt.savefig(path2+str(files[i])+'_p53_ridge_'+str(count)+'.svg')
            plt.show()

        
    
