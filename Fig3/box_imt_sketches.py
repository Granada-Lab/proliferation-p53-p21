#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 13:41:49 2023

@author: nicagutu
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 22})    

path = 'Output/'

# mean1, std1 = 24, 6
# mean2, std2 = 30, 6

# samples1 = np.random.normal(mean1, std1, 10000000)
# samples2 = np.random.normal(mean2, std2, 10000000)

# fig = plt.figure(figsize=(10,8))
# sns.kdeplot(data=samples1, label='Untreated', color='tab:blue')
# sns.kdeplot(data=samples2, label='Treated', color='tab:brown')
# # sns.histplot(samples1, bins=100, kde=True, stat='density', label='Untreated', color='tab:blue')
# # sns.histplot(samples2, bins=100, kde=True, stat='density', label='Treated', color='tab:brown')
# plt.xlabel('IMT [hours]')
# plt.xlim([5,45])
# plt.legend()
# plt.savefig(path+'IMT_distributions_diff_means.svg')
# plt.show()

# time = np.linspace(0, 120, 240)
# y1 = np.exp(0.03*time)
# y2 = np.exp(0.02*time)

# fig = plt.figure(figsize=(10,8))
# plt.plot(time, y1, label='Untreated', color='tab:blue')
# plt.plot(time, y2, label='Treated', color='tab:brown')
# plt.xlabel('Time [hours]')
# plt.ylabel('Population growth')
# plt.legend()
# plt.savefig(path+'Exponential_growth.svg')
# plt.show()

num_divisions = [0,1,2,3,4,5]
IMT1 = [40,36,32,28,24,20]
IMT2 = [24,24,24,24,24,24]

fig = plt.figure(figsize=(10,8))
plt.plot(num_divisions, IMT1, '-o', label='1st scenario', color='tab:blue')
plt.plot(num_divisions, IMT2, '-o', label='2nd scenario', color='tab:brown')
plt.xlabel('Total number of divisions')
plt.ylabel('IMT [hours]')
plt.ylim([15,45])
plt.legend()
plt.savefig(path+'IMT_divisions_sketch.svg')
plt.show()






