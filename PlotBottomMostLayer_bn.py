# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:04:24 2022

@author: baongoc.thai
"""

import pandas as pd
import os
import matplotlib.pyplot as plt


path = r'S:\01_PROJECTS\H2I-C2020-015_ERM_PV-Kranji\Working-Documents\2_Models\GCM\WAQ results\2019map'
os.chdir(path)
cellworking = r'S:\01_PROJECTS\H2I-C2020-015_ERM_PV-Kranji\Working-Documents\2_Models\GCM\WAQ results\cell_working.xlsx'

#%% NPV
data = pd.read_csv('NPV.csv')

# prepare data for middle segment (1.5-2.5m depth)
mid_depth = pd.read_excel(cellworking,sheet_name='raw')
mid_depth = mid_depth[(mid_depth['z coordinate']<=-1.5) & (mid_depth['z coordinate']>=-2.5)]
mid_depth_segment = list(mid_depth['Segment'])
data_mid_depth_segment = data[data.columns.intersection(mid_depth_segment)]
data_mid_depth_segment.index = data['Location:']
data_mid_depth_mean = pd.DataFrame(data_mid_depth_segment.mean(axis=0))
data_mid_depth_mean_location = pd.merge(data_mid_depth_mean, mid_depth[['x coordinate', 'y coordinate','z coordinate','Segment']],
                              how="inner", left_on=data_mid_depth_mean.index, right_on='Segment')
data_mid_depth_mean_location.to_csv('mid-depth_segment_NPV.csv')

# Plot for middle segments
plt.figure(figsize = (15,12))
plt.scatter(data_mid_depth_mean_location['x coordinate'], data_mid_depth_mean_location['y coordinate'], c=data_mid_depth_mean_location[0], 
            cmap='Set2'
            ,vmin=2, vmax=10
            )
plt.colorbar().set_label('DO (mg/L)')
plt.rcParams.update({'font.size': 25})
plt.xticks(color='w')
plt.yticks(color='w')
plt.title('2019 NPV (mid-depth)')
#plt.show()
plt.savefig("2019_NPV_DO_mid-depth.png")
plt.close()

# prepare data for bottom segments
segment_bottom_most = pd.read_excel(cellworking,sheet_name='bottom-most')
bottom_segment = list(segment_bottom_most['Segment'])
data_bottom_segment = data[data.columns.intersection(bottom_segment)]
data_bottom_segment.index = data['Location:']
data_bottom_mean = pd.DataFrame(data_bottom_segment.mean(axis=0))
data_bottom_mean_location = pd.merge(data_bottom_mean, segment_bottom_most[['x coordinate', 'y coordinate','z coordinate','Segment']],
                              how="inner", left_on=data_bottom_mean.index, right_on='Segment')
#Exclude locations with only surface layer
data_bottom_mean_location = data_bottom_mean_location[data_bottom_mean_location['z coordinate']<-1]
data_bottom_mean_location.to_csv('bottom-most_segment_NPV.csv')

# Plot for bottom segments
plt.figure(figsize = (15,12))
plt.scatter(data_bottom_mean_location['x coordinate'], data_bottom_mean_location['y coordinate'], c=data_bottom_mean_location[0], 
            cmap='Set2'
            ,vmin=2, vmax=10
            )
plt.colorbar().set_label('DO (mg/L)')
plt.rcParams.update({'font.size': 25})
plt.xticks(color='w')
plt.yticks(color='w')
plt.title('2019 NPV (bottom)')
#plt.show()
plt.savefig("2019_NPV_DO_bottom-most.png")
plt.close()

#%% PV
data = pd.read_csv('PV.csv')

# prepare data for middle segment (1.5-2.5m depth)
mid_depth = pd.read_excel(cellworking,sheet_name='raw')
mid_depth = mid_depth[(mid_depth['z coordinate']<=-1.5) & (mid_depth['z coordinate']>=-2.5)]
mid_depth_segment = list(mid_depth['Segment'])
data_mid_depth_segment = data[data.columns.intersection(mid_depth_segment)]
data_mid_depth_segment.index = data['Location:']
data_mid_depth_mean = pd.DataFrame(data_mid_depth_segment.mean(axis=0))
data_mid_depth_mean_location = pd.merge(data_mid_depth_mean, mid_depth[['x coordinate', 'y coordinate','z coordinate','Segment']],
                              how="inner", left_on=data_mid_depth_mean.index, right_on='Segment')
data_mid_depth_mean_location.to_csv('mid-depth_segment_PV.csv')

# Plot for middle segments
plt.figure(figsize = (15,12))
plt.scatter(data_mid_depth_mean_location['x coordinate'], data_mid_depth_mean_location['y coordinate'], c=data_mid_depth_mean_location[0], 
            cmap='Set2'
            ,vmin=2, vmax=10
            )
plt.colorbar().set_label('DO (mg/L)')
plt.rcParams.update({'font.size': 25})
plt.xticks(color='w')
plt.yticks(color='w')
plt.title('2019 PV (mid-depth)')
#plt.show()
plt.savefig("2019_PV_DO_mid-depth.png")
plt.close()

# prepare data for bottom segments
segment_bottom_most = pd.read_excel(cellworking,sheet_name='bottom-most')
bottom_segment = list(segment_bottom_most['Segment'])
data_bottom_segment = data[data.columns.intersection(bottom_segment)]
data_bottom_segment.index = data['Location:']
data_bottom_mean = pd.DataFrame(data_bottom_segment.mean(axis=0))
data_bottom_mean_location = pd.merge(data_bottom_mean, segment_bottom_most[['x coordinate', 'y coordinate','z coordinate','Segment']],
                              how="inner", left_on=data_bottom_mean.index, right_on='Segment')
#Exclude locations with only surface layer
data_bottom_mean_location = data_bottom_mean_location[data_bottom_mean_location['z coordinate']<-1]
data_bottom_mean_location.to_csv('bottom-most_segment_PV.csv')

# Plot for bottom segments
plt.figure(figsize = (15,12))
plt.scatter(data_bottom_mean_location['x coordinate'], data_bottom_mean_location['y coordinate'], c=data_bottom_mean_location[0], 
            cmap='Set2'
            ,vmin=2, vmax=10
            )
plt.colorbar().set_label('DO (mg/L)')
plt.rcParams.update({'font.size': 25})
plt.xticks(color='w')
plt.yticks(color='w')
plt.title('2019 PV (bottom)')
#plt.show()
plt.savefig("2019_PV_DO_bottom-most.png")
plt.close()

