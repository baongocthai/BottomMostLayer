# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:28:10 2022

@author: baongoc.thai
"""

#%% Annual average from Monthly mean (-stat.map files) 
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

path = r'S:\01_PROJECTS\H2I-C2020-015_ERM_PV-Kranji\Working-Documents\2_Models\GCM\WAQ results\Monthly-mean'
os.chdir(path)
cellworking = r'S:\01_PROJECTS\H2I-C2020-015_ERM_PV-Kranji\Working-Documents\2_Models\GCM\WAQ results\cell_working.xlsx'

year = [2019, 2030, 2040, 2050]
scenario = ['NPV', 'PV']
parameters = ['Chlfa','NH4','OXY','SS','TOC','TotN','TotP']
vmin_lim = [30, 0.03, 4, 10, 7, 0.7, 0.05]
vmax_lim = [70, 0.19, 8, 50, 15, 1.5, 0.13]

#mid-depth
raw = pd.read_excel(cellworking,sheet_name='raw')
raw.index = raw['Segment']
mid_depth = raw[(raw['z coordinate']<=-2) & (raw['z coordinate']>=-3)]

for i in range(len(scenario)):
    for j in range(len(year)):
        data = pd.read_csv('MonthlyMean-WAQ' + str(year[j]) + str(scenario[i]) + '.csv',skiprows=[i for i in range(0,4)])
        
        for k in range(len(parameters)):
            data_filter = data[data.filter(like=parameters[k]).columns]
            data_filter = data_filter.replace({-999:np.nan})
            data_filter.index = data['Parameter:']
            data_filter_dropna = data_filter.dropna()
            data_filter_dropna_annual_mean = pd.DataFrame(data_filter_dropna.mean(axis=1))
            
            mid_depth_annual_mean = pd.merge(mid_depth, data_filter_dropna_annual_mean,left_index=True, right_index=True)
            mid_depth_annual_mean = mid_depth_annual_mean[['x coordinate','y coordinate','z coordinate',0]]
            
            # Plot for middle segments
            plt.figure(figsize = (15,12))
            plt.scatter(mid_depth_annual_mean['x coordinate'], mid_depth_annual_mean['y coordinate'], c=mid_depth_annual_mean[0], 
                        cmap='Set2'
                        ,vmin=vmin_lim[k], vmax=vmax_lim[k]
                        )
            plt.colorbar().set_label(parameters[k])
            plt.rcParams.update({'font.size': 25})
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.title(str(year[j])+ ' ' + str(scenario[i]) + ' ' + parameters[k] + ' (mid-depth)')
            #plt.show()
            plt.savefig('Figures-mid-depth\\' + str(year[j])+ ' ' + str(scenario[i]) + ' ' + parameters[k] + ' (mid-depth).png')
            plt.close()
            print (parameters[k])
        
        print (year[j])
    
    print (scenario[i])
    
#bottom-most
segment_bottom_most = pd.read_excel(cellworking,sheet_name='bottom-most')
segment_bottom_most.index = segment_bottom_most['Segment']
segment_bottom_most = segment_bottom_most[segment_bottom_most['z coordinate'] <-1]

for i in range(len(scenario)):
    for j in range(len(year)):
        data = pd.read_csv('MonthlyMean-WAQ' + str(year[j]) + str(scenario[i]) + '.csv',skiprows=[i for i in range(0,4)])
        
        for k in range(len(parameters)):
            data_filter = data[data.filter(like=parameters[k]).columns]
            data_filter = data_filter.replace({-999:np.nan})
            data_filter.index = data['Parameter:']
            data_filter_dropna = data_filter.dropna()
            data_filter_dropna_annual_mean = pd.DataFrame(data_filter_dropna.mean(axis=1))
            
            mid_depth_annual_mean = pd.merge(segment_bottom_most, data_filter_dropna_annual_mean,left_index=True, right_index=True)
            mid_depth_annual_mean = mid_depth_annual_mean[['x coordinate','y coordinate','z coordinate',0]]
            
            # Plot for middle segments
            plt.figure(figsize = (15,12))
            plt.scatter(mid_depth_annual_mean['x coordinate'], mid_depth_annual_mean['y coordinate'], c=mid_depth_annual_mean[0], 
                        cmap='Set2'
                        ,vmin=vmin_lim[k], vmax=vmax_lim[k]
                        )
            plt.colorbar().set_label(parameters[k])
            plt.rcParams.update({'font.size': 25})
            plt.xticks(color='w')
            plt.yticks(color='w')
            plt.title(str(year[j])+ ' ' + str(scenario[i]) + ' ' + parameters[k] + ' (bottom-most)')
            #plt.show()
            plt.savefig('Figures-bottom-most\\' + str(year[j])+ ' ' + str(scenario[i]) + ' ' + parameters[k] + ' (bottom-most).png')
            plt.close()
            print (parameters[k])
        
        print (year[j])
    
    print (scenario[i])