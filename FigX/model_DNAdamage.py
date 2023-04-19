#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:56:12 2023

@author: nicagutu
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 24})
plt.rcParams['svg.fonttype'] = 'none'

path = 'Output/'

#%%Logistic function

N = 30

##Fixed parameter and changing center
c = 1 #maximum y
k = 20 #exp parameter ##all cells have the same threshold

max_center = 5
x = np.linspace(-1*max_center, max_center*2, 1000)
x0_values = np.linspace(0, max_center, N)
y_sum = np.zeros_like(x)

#Single-cell level 
fig = plt.figure(figsize=(12,10))
for i in range(N):
    x0 = x0_values[i] #np.random.uniform(0, max_center) #
    y = - c / (1 + np.exp(-k*(x-x0))) #Logistic function
    y_sum += y #Sum function
    plt.plot(x, y, alpha=.5)
plt.xlim([-2,7])
plt.xticks([])
plt.yticks([])
plt.xlabel('Endogenous DNA damage')
plt.ylabel('Single-cell proliferation')
plt.savefig(path+'Single_cell_functions_model.svg')
plt.show()

#Population   
fig = plt.figure(figsize=(12,10))
plt.plot(x, y_sum)
plt.xticks([])
plt.yticks([])
plt.xlabel('Endogenous DNA damage')
plt.ylabel('Population proliferation')
plt.savefig(path+'Population_model.svg')
plt.show()

##Fixed center and changing parameter
# k_values = np.linspace(0.01, 1, N)

# def sigmoid_sum(x):
#     y = 0
#     for k in k_values:
#         y += 1 / (1 + np.exp(-k * x))
#     return y / N

# x_values = np.linspace(-10, 10, 1000)
# y_values = sigmoid_sum(x_values)

# #Single-cell level 
# fig = plt.figure(figsize=(12,10))
# for k in k_values:
#     plt.plot(x_values, 1 / (1 + np.exp(-k * (x_values))), alpha=.5)
# plt.xlabel('Endogenous DNA damage')
# plt.ylabel('Cells')
# plt.show()

# #Population 
# fig = plt.figure(figsize=(12,10))
# plt.plot(x_values, y_values)
# plt.xlabel('Endogenous DNA damage')
# plt.ylabel('Proliferation')
# plt.show()



 #%%Heaviside function
# # Define the number of Heaviside functions
# N = 10

# # Define the center values of the Heaviside functions
# step_centers = np.linspace(0, 1, N)

# # Define the function as a sum of Heaviside functions
# def step_sum(x):
#     y = 0
#     for center in step_centers:
#         y += np.heaviside(x - center, 1)
#     return y

# x_values = np.linspace(0, 1, 1000)
# y_values = step_sum(x_values)

# fig = plt.figure(figsize=(12,10))
# plt.plot(x_values, y_values)
# plt.show()

