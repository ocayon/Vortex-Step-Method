# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 20:53:40 2022

@author: oriol
"""

import numpy as np
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt
from csv import writer

## generate the data and plot it for an ideal normal curve

## x-axis for the plot
x_data = np.arange(0, 180, 2)

## y-axis as the gaussian
mean = 90
std_dev = 34
y_data = stats.norm.pdf(x_data, loc=mean, scale=std_dev)

y_data =np.around(y_data,4)

coeffs = np.zeros((len(x_data),4))



coeffs[:,0] = x_data
coeffs[:,2] = y_data*170

coeffs[:,1] = 0.02187*x_data**2-1.087*x_data+14.40
coeffs[:,1] = 0.01333*x_data**2-0.6967*x_data+10
# coeffs[:,1] = 449/33600*x_data**2-11621/16800*x_data+3413/350

row_contents = coeffs[10:17,:]
with open('./polars/clarky_maneia.csv', "a", newline="") as f_object:
  
    # Pass this file object to csv.writer()
    # and get a writer object
    writer_object = writer(f_object)
  
    # Pass the list as an argument into
    for i in range(len(row_contents)):
        # the writerow()
        writer_object.writerow(row_contents[i])
  
    #Close the file object
    f_object.close()
    
    
data_airf = np.loadtxt(r'./polars/clarky_maneia.csv',delimiter = ',')

plt.figure()
plt.plot(data_airf[:,0],data_airf[:,1])
plt.figure()
plt.plot(data_airf[:,0],data_airf[:,2])
