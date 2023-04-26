# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 13:31:36 2023

@author: baongoc.thai
"""

# =============================================================================
# This script plots contour plot for water quality from profiler data
# =============================================================================
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

directory = r'S:\01_PROJECTS\H2I-C2020-015_ERM_PV-Kranji\Working-Documents\Received data\10 PUB Data\07_Online WQ Data_Kranji 2_Jan 2018 to Dec 2019_for EDB'
os.chdir(directory)

#%% Main block: Extract results from interested stations
list_file = glob.glob('*_BN.csv')
for i in range(len(list_file)):
    filename = list_file[i]
    data = pd.read_csv(filename,sep=',',index_col=0)
    data.index = pd.to_datetime(data.index, format='%d/%m/%Y %H:%M')
    data.columns = [0.5,1,1.5,2,4,6]
    data_selected = data[data.index.year == 2019]
    data_selected_hourly = data_selected.resample('H').mean()
    
    x = data_selected_hourly.index
    y = data.columns
    px_values = data_selected_hourly.transpose()
    
    fig, ax = plt.subplots()
    
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'large'
    plt.rcParams["figure.figsize"] = (18, 6)
    
    CS = ax.contourf(x, y, px_values, cmap ='rainbow')
    plt.gca().invert_yaxis()
    fig.colorbar(CS)
    plt.ylabel(filename[0:-7])
    plt.xlabel("Time")
    print (filename)
    plt.savefig(filename[0:-7]+'_profiler.png', bbox_inches='tight',dpi=600)
    plt.close()





