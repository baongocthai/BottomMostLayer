# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 10:53:41 2023

@author: baongoc.thai
"""
# =============================================================================
# This script 
#   - Plot profiler data: temperature profile & temperature difference between top & bottom
#   - Interpolate modelled water temperature data based on same depth as observed in profiler data
#   - Plot temperature difference for top & bottom modelled temperature
#   - Integrate heat flux terms for the entire reservoir surface
# =============================================================================
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import datetime
import numpy as np
# from windrose import WindroseAxes
from scipy import interpolate
from scipy.stats import pearsonr
from pymatreader import read_mat
import re
from functools import reduce
import matplotlib.patches as mpatches
import matplotlib

#%% Function to import data
def ReadData(filename):
    pd1 = pd.read_csv(filename,encoding= 'unicode_escape')
    pd1.index = pd1.pop('date and time')
    pd1.index = pd.to_datetime(pd1.index,format = "%Y-%m-%d %H:%M:%S")
    pd1 = pd1.replace(r'^\s*$', np.nan, regex=True) # replace blanks with nan
    return pd1

#%% Import WAQ results function:
def data_import(year, scenario):
    df = pd.read_csv(str(year) + "_" + scenario + ".csv",dayfirst=True,skiprows=range(0,3))
    columns = [text.split(".")[0] for text in df.columns]
    df.columns = columns
    df.columns = df.columns + "_" + df.iloc[0] #Parameter + Locations
    df = df.rename(columns = {'Parameter:_Location:':'Date'})
    df = df.iloc[1:] #drop first row
    df.index = df.pop('Date')
    df = df.astype(float)
    df[df < 0] = 0 #Change all negative values to 0
    try:
        df.index = pd.to_datetime(df.index, format = "%Y/%m/%d %H:%M:%S")
    except ValueError:
        df.index = pd.to_datetime(df.index, format = "%d/%m/%Y %H:%M")
    return df
#%% Function to interpolate values with depth
def temperature_interpolate(dataframe, depth_model, depth_obs):
    model_raw = dataframe
    model_raw.columns = depth_model
    obslevel= depth_obs
    model = pd.DataFrame()
    
    for m in range(len(model_raw)):
        model_raw_each = model_raw.iloc[[m]]
        model_row_each_1 = model_raw_each.dropna(axis=1)
        x = np.array(model_row_each_1.columns)
        y = np.array(model_row_each_1.iloc[0])
        f = interpolate.interp1d(x,y,kind='linear',fill_value='extrapolate')
        model_each = f(obslevel)
        model = pd.concat([model, pd.DataFrame(model_each).transpose()])
    
    model.index = dataframe.index
    model.columns = depth_obs
    return model

#%% Function to process mat file (map result from D3D-FLOW) for Heat Flux terms - integrate for whole reservoir
def ProcessMatFile(filename, grid_surface_area):
    data = read_mat(filename)['data']['Val']
    # data_whole = data[0:,1:,1:]*grid_surface_area
    data_whole = data*grid_surface_area
    TotalArea = np.nansum(grid_surface_area)
    data_whole_ts = np.nansum(np.nansum(data_whole, axis=1), axis=1)/TotalArea
    print(filename)
    return data_whole_ts
#%% Function to process mat file (map result from D3D-FLOW) for Heat Flux terms - average annual
def ProcessMatFileAnnualAvg(filename, grid_surface_area):
    data = read_mat(filename)['data']['Val']
    # data_whole = data[0:,1:,1:]*grid_surface_area
    data_whole = data*grid_surface_area
    data_avg = np.nanmean(data_whole, axis=0)
    return data_avg

#%% Plot Dt NPV-PV (for deltaT criteria)
def plot_temp_diff(temp_diff, year, area, para):   
     plt.rc('xtick', labelsize=18)
     plt.rc('ytick', labelsize=18)
     fig, ax = plt.subplots()
     fig.set_size_inches(9, 8)
     temp_diff.boxplot(notch=True, widths = 0.4,
                # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
                # capprops=dict(color="steelblue"),
                whiskerprops=dict(color='gray',linewidth=1.5),
                flierprops=dict(color='gray',markersize=1.5),
                medianprops=dict(color="black"),
                # patch_artist = True,
                )
     plt.xticks([1, 2, 3], list(area.values()), size = 18)
     plt.axhline(y = 0.2, color = 'orange', linewidth=2.5) 
     plt.grid(visible=True,linestyle='-')
     plt.ylim([-0.25, 0.75])
     ax.tick_params(direction="in")
     plt.title('Year ' + str(year) + ", [PV-NPV]",size = 20)
     plt.ylabel(para['Temp'],size = 20)
     plt.savefig('Temp'+" Diff_"+str(year)+"_annual.png", bbox_inches='tight')
     plt.close()
#%% Plot - no criteria line (for guidelines)
def plot_nocriteria(npv, pv, year, parameters, area):
    for i in range(len(parameters)):
        df_temp1 = npv.filter(regex=parameters[i])
        df_temp2 = pv.filter(regex=parameters[i])
        df = df_temp1.merge(df_temp2, how='outer', left_index=True, right_index=True, suffixes=('_NPV', '_PV'))
        iqr = np.percentile(df, 75) - np.percentile(df, 25)
        maximum = df.max()+0.5*iqr
        minimum = df.min()-0.5*iqr
        
        #Plot
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        fig, ax = plt.subplots()
        fig.set_size_inches(18, 8)
        axes = df.boxplot(notch=True, widths = 0.4,
                   # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
                   # capprops=dict(color="steelblue"),
                   whiskerprops=dict(color='gray',linewidth=1.5),
                   flierprops=dict(color='gray',markersize=1.5),
                   medianprops=dict(color="black"),
                   patch_artist = True,
                   )
        for k in range(len(df.columns)):
            if k<=2:
                axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='-',color='steelblue',linewidth=2.5)
                axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
                # axes.findobj(matplotlib.patches.Patch)[i].set_edgecolor('steelblue')
            else:
                axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='--',color='olivedrab',linewidth=2.5)
                axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
                # axes.findobj(matplotlib.patches.Patch)[i].set_edgecolor('olivedrab')

        # Criteria line & legends
        # if parameters[i] in criteria_text:
        #     ax.axhline(y=criteria_value[parameters[i]], color='orange', linewidth=3, label = criteria_text[parameters[i]])
        #     handles, labels = ax.get_legend_handles_labels()
        #     patch1 = mpatches.Patch(edgecolor='steelblue', facecolor="white", label='NPV')
        #     patch2 = mpatches.Patch(edgecolor='olivedrab', facecolor="white", linestyle='--', label='PV')
        #     handles.append(patch1)
        #     handles.append(patch2)
        #     handles_arr = [handles[i] for i in [1,2,0]]
        # else:
        #     patch1 = mpatches.Patch(edgecolor='steelblue', facecolor="white", label='NPV')
        #     patch2 = mpatches.Patch(edgecolor='olivedrab', facecolor="white", linestyle='--', label='PV')
        #     handles_arr = [patch1, patch2]
        try:
            if parameters[i] == 'Temp':
                plt.ylim([minimum.min(), maximum.max()])
            elif parameters[i] == 'OXY':
                plt.ylim([-1, 9])
            elif maximum.max() < criteria_value[parameters[i]]:
                plt.ylim([minimum.min(), criteria_value[parameters[i]]])
            elif minimum.min() > criteria_value[parameters[i]]:
                plt.ylim([criteria_value[parameters[i]], maximum.max()])
        except KeyError:
            plt.ylim([minimum.min(), maximum.max()])    
        
        # if parameters[i] == 'OXY':
        #     ax.legend(handles = handles_arr, loc='lower left',fontsize=20)
        # else:
        #     ax.legend(handles = handles_arr, loc='upper left',fontsize=20)   
        
        plt.grid(visible=True,linestyle='-')
        ax.axvline(x=3.5, color='black', linestyle=':', linewidth=2)
        xlim_original = ax.get_xlim()
        plt.xlim(xlim_original)
        ax.tick_params(direction="in")
        # plt.text(1.5, maximum.max(), 'Without FPV', size = 25, color = 'steelblue')
        # plt.text(5, maximum.max(), 'With FPV', size = 25, color = 'olivedrab')
        plt.title('Year ' + str(year) + ' (Depth Average)',size = 35)
        plt.xticks([1, 2, 3, 4, 5, 6], list(area.values())*2, size = 20)
        plt.ylabel(para[parameters[i]],size = 20)       
        plt.savefig(parameters[i]+"_"+str(year)+"_annual.png", bbox_inches='tight')
        print(parameters[i])
        plt.close()
#%% Main block: import data
directory = Path(r'C:\Users\923627\OneDrive - Royal HaskoningDHV\Desktop\NgocNgoc\Work\LSR FPV EIA\Stratification')
os.chdir(directory)

# Profiler data depth
depth_obs = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5])
profiler_data = 'LSR_TemperatureProfiler_2019.csv'

# Model data depth
depth_mod = np.array([6.5, 6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 0.6]) #Change Vertical layer for 1st layer
model_data = 'temperature_SE503_2019NPV_VerticalGeometry.csv'

#Import temperature profile
temperature_profiler = pd.read_csv(profiler_data)
temperature_profiler.index = temperature_profiler.pop('date')
temperature_profiler.index = pd.to_datetime(temperature_profiler.index, format = '%d/%m/%Y %H:%M')
temperature_profiler = temperature_profiler.sort_index()
temperature_profiler.columns = depth_obs

# Interpolate by time for temperature profiler data
temperature_profiler_hourly = temperature_profiler.resample('H').mean()

# Modelled temperature
temp_mod = ReadData(model_data)
temp_mod.columns = depth_mod
temp_mod = temp_mod.astype('float')

#%% Main block: Interpolate modelled temperature at specified depth (same as in profiler data)
temp_inter = temperature_interpolate(temp_mod, depth_mod, depth_obs)

for n in range(len(depth_obs)):
    plt.scatter(temperature_profiler_hourly.index,temperature_profiler_hourly[depth_obs[n]], s=1,  color='black',label="Observation")
    plt.plot(temp_inter.index, temp_inter[depth_obs[n]], color='blue', linewidth=1, label='Model')
    plt.xlim([datetime.datetime(2019, 1, 1), datetime.datetime(2020, 1, 1)])
    plt.ylim([25,35])
    plt.ylabel("Temperature (oC)")
    plt.title("Temperature at Profiler (" + str(depth_obs[n]) +"m depth)")
    plt.legend(loc='upper right',fontsize=10)
    plt.rcParams.update({'font.size': 15})
    figure = plt.gcf() # get current figure
    figure.set_size_inches(16,3.5)
    plt.savefig('Temperature_LSR_VerticalGeometry\\'+"Temperature at Profiler (" + str(depth_obs[n]) +"m depth).png", bbox_inches='tight',dpi=300)
    plt.close()
    
# Compare profiler and modelled results
temp_inter_2019 = temp_inter.loc[temp_inter.index.year == 2019]
temperature_profiler_hourly_comparison = temperature_profiler_hourly.loc[temp_inter_2019.index]
temperature_profiler_hourly_comparison = temperature_profiler_hourly_comparison.dropna()
temp_inter_2019 = temp_inter_2019.loc[temperature_profiler_hourly_comparison.index]
temp_inter_2019 = temp_inter_2019.sort_index()
temperature_profiler_hourly_comparison = temperature_profiler_hourly_comparison.sort_index()

#Compare with obs & modelled results
difference = abs(temp_inter_2019 - temperature_profiler_hourly_comparison)
mae = difference.mean(axis = 0, skipna = True)
mse = (difference**2).mean(axis = 0, skipna = True)
rmse = (difference**2).mean(axis = 0, skipna = True)**0.5
pearson = pd.Series([pearsonr(temp_inter_2019[col],temperature_profiler_hourly_comparison[col])[0] for col in depth_obs])
pearson.index = rmse.index
std_modelled = temp_inter_2019.std()
mean_modelled = temp_inter_2019.mean()
std_temperature_profiler_hourly_comparison = temperature_profiler_hourly_comparison.std()
mean_temperature_profiler_hourly_comparison = temperature_profiler_hourly_comparison.mean()
stats_results = pd.concat([mae, mse, rmse, pearson, std_modelled, mean_modelled, std_temperature_profiler_hourly_comparison, mean_temperature_profiler_hourly_comparison], axis=1)
stats_results.columns = ['mae','mse','rmse','pearson r','modelled std','modelled mean','obs std','obs mean']
stats_results.to_csv('Vertical temperature profile stats at Profiler_VerticalGeometry.csv')

#%% Main block: Plot temperature contour - Modelled data
levels = np.linspace(25.3,33,12)
temp_mod_na = temp_mod.dropna(axis=1, how='all')
temp_mod_na_2019 = temp_mod_na[temp_mod_na.index.year == 2019]

for m in range(0,12,3):
    temp_mod_month = temp_mod_na_2019.loc[temp_mod_na_2019.index.month.isin([m+1,m+2,m+3])]

    #Plot temperature difference
    x = temp_mod_month.index
    y = temp_mod_month.columns
    px_values = temp_mod_month.transpose()
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'large'
    plt.rcParams["figure.figsize"] = (16, 4)
    CS = ax.contourf(x, y, px_values, levels = levels, cmap ='rainbow')
    fig.colorbar(CS)
    plt.gca().invert_yaxis()
    plt.ylabel('Water level wrt surface (m)', size = 14)
    plt.title('Modelled temperature at profiler location during month ' + str(m+1)+ ' to ' +str(m+3))
    plt.savefig('Temperature_LSR_VerticalGeometry\\'+'Modelled temperature at profiler location during month ' + str(m+1) + ' to ' +str(m+3) +  '.png', bbox_inches='tight',dpi=300)
    plt.close()
#%% Main block: Plot temperature contour - Profiler data
# levels = np.linspace(26.4,33,12)
levels = np.linspace(25.3,33,12)

for m in range(0,12,3):
    temp_mod_month = temperature_profiler_hourly_comparison.loc[temperature_profiler_hourly_comparison.index.month.isin([m+1,m+2,m+3])]

    #Plot temperature difference
    x = temp_mod_month.index
    y = temp_mod_month.columns
    px_values = temp_mod_month.transpose()
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 'large'
    plt.rcParams['figure.titlesize'] = 'large'
    plt.rcParams["figure.figsize"] = (16, 4)
    CS = ax.contourf(x, y, px_values, levels = levels, cmap ='rainbow')
    fig.colorbar(CS)
    plt.gca().invert_yaxis()
    plt.ylabel('Water level wrt surface (m)', size = 14)
    plt.title('Measured temperature at profiler location during month ' + str(m+1)+ ' to ' +str(m+3))
    plt.savefig('Temperature_LSR_VerticalGeometry\\'+'Measured temperature at profiler location during month ' + str(m+1) + ' to ' +str(m+3) +  '.png', bbox_inches='tight',dpi=300)
    plt.close()
#%% Main block: Read mat file - Whole reservoir time series heat flux terms
grid_file = r"C:\Users\923627\OneDrive - Royal HaskoningDHV\Desktop\NgocNgoc\Work\LSR FPV EIA\Stratification\grid cell surface area.mat"

# Extract surface grid cell areas from mat file
grid_data = read_mat(grid_file)
SurfaceArea = grid_data['data']['Val']
Grid_x =  grid_data['data']['X']
Grid_y =  grid_data['data']['Y']
# Grid_z =  grid_data['data']['Z']
TotalArea = np.nansum(SurfaceArea)

year = 2019 
scenario = 'FPV'
directory = Path(r'C:\Users\923627\OneDrive - Royal HaskoningDHV\Desktop\NgocNgoc\Work\LSR FPV EIA\Stratification\HeatFlux_VerticalGeometry_'+str(year)+scenario)
os.chdir(directory)

import glob
files = glob.glob("*.mat")
column_names = [file.split('.')[0] for file in files]
LSR = []

for file in files:
    data = ProcessMatFile(file, SurfaceArea)
    LSR.append(data)

LSR_df = pd.DataFrame(LSR).transpose()
LSR_df.columns = column_names
LSR_df.to_csv('WholeLSR_heatflux_VerticalGeometry_'+str(year)+scenario+'.csv')


#%% Annual boxplot - temperature absolute & diff (bottom-water)
directory = r"C:\Users\923627\OneDrive - Royal HaskoningDHV\Desktop\NgocNgoc\Work\LSR FPV EIA\Stratification"
os.chdir(directory)
area = {'LSR-W_Bottom' : 'Whole Reservoir\nBottom water'}
# area = {'LSR-W_Bottom_3layers' : 'Whole Reservoir\nBottom water (3 layers)'}

para = {"Chlfa": "Chlorophyll-a (ug/L)", "OXY": "Dissolved oxygen (mg/L)",
        "PO4": "Phosphate (mg/L)",
        "SS": "Suspended solid (mg/L}", "Temp": "Diff. Water temperature (deg C)",
        "TOC": "Total organic carbon (mg/L)", "TotN": "Total nitrogen (mg/L)",
        "TotP": "Total phosphorus (mg/L)", "NH4": "Ammonia (mg/L)"}
parameters = list(para.keys())
criteria_text = {"Chlfa": "50 ug/L", "OXY": "3 mg/L", "TOC": "10 mg/L", "TotN": "1 mg/L",
                 "TotP": "0.06 mg/L", "NH4": "0.5 mg/L"} 
criteria_value = {"Chlfa": 50, "OXY": 3, "TOC": 10, "TotN": 1, "TotP": 0.06, "NH4": 0.5}  

year = 2019

df2019_npv = data_import(str(year), 'NPV_day')
df2019_pv = data_import(str(year), 'FPV_day_PV3')

Temp_col = df2019_npv.columns[df2019_npv.columns.str.contains(pat = 'Temp')] 
selected_areas_col = [col for col in Temp_col if col.split("_", 1)[1] in area.keys()]

#Only temperature
df2019_npv = data_import(str(year), 'NPV_day')[selected_areas_col]
df2019_pv = data_import(str(year), 'FPV_day_PV3')[selected_areas_col]

df_Temp_diff_2019 = df2019_pv - df2019_npv
columns = df_Temp_diff_2019.columns

Temp_diff_exceedance_2019 = []

#2019 difference
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
fig, ax = plt.subplots()
fig.set_size_inches(4, 8.5)
df_Temp_diff_2019.boxplot(notch=True, widths = 0.4,
           # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
           # capprops=dict(color="steelblue"),
           whiskerprops=dict(color='gray',linewidth=1.5),
           flierprops=dict(color='gray',markersize=1.5),
           medianprops=dict(color="black"),
           # patch_artist = True,
           )
plt.xticks([1], list(area.values()), size = 18)
plt.axhline(y = 0.2, color = 'orange', linewidth=2.5) 
plt.grid(visible=True,linestyle='-')
plt.ylim([-0.25, 0.75])
ax.tick_params(direction="in")
plt.title('Year ' + str(year) + ", [PV-NPV]",size = 20)
plt.ylabel(para['Temp'],size = 20)
plt.savefig('Temp'+" Diff_BottomWater_PV3_"+str(year)+"_annual.png", bbox_inches='tight')
plt.close()

#2019 absolute
df = df2019_npv.merge(df2019_pv, how='outer', left_index=True, right_index=True, suffixes=('_NPV', '_PV'))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
fig, ax = plt.subplots()
fig.set_size_inches(8, 8.5)
axes = df.boxplot(notch=True, widths = 0.4,
           # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
           # capprops=dict(color="steelblue"),
           whiskerprops=dict(color='gray',linewidth=1.5),
           flierprops=dict(color='gray',markersize=1.5),
           medianprops=dict(color="black"),
           patch_artist = True,
           )
for k in range(len(df.columns)):
    if k<=0:
        axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='-',color='steelblue',linewidth=2.5)
        axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
        # axes.findobj(matplotlib.patches.Patch)[i].set_edgecolor('steelblue')
    else:
        axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='--',color='olivedrab',linewidth=2.5)
        axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
plt.grid(visible=True,linestyle='-')
xlim_original = ax.get_xlim()
plt.xlim(xlim_original)
ax.tick_params(direction="in")
# plt.text(1.5, maximum.max(), 'Without FPV', size = 25, color = 'steelblue')
# plt.text(5, maximum.max(), 'With FPV', size = 25, color = 'olivedrab')
plt.title('Year ' + str(year),size = 35)
plt.xticks([1, 2], list(area.values())*2, size = 20)
plt.ylabel('Water temperature (deg C)',size = 20)       
plt.savefig('Temp'+"_BottomWater_PV3_"+str(year)+"_annual.png", bbox_inches='tight')
plt.close()

#%% Annual boxplot - DO absolute & diff (bottom-water)
directory = r"C:\Users\923627\OneDrive - Royal HaskoningDHV\Desktop\NgocNgoc\Work\LSR FPV EIA\Stratification"
os.chdir(directory)
area = {'LSR-W_Bottom' : 'Whole Reservoir\nBottom water'}
# area = {'LSR-W_bottom' : 'Whole Reservoir\nBottom water'}

para = {"Chlfa": "Chlorophyll-a (ug/L)", "OXY": "Dissolved oxygen (mg/L)",
        "PO4": "Phosphate (mg/L)",
        "SS": "Suspended solid (mg/L}", "Temp": "Diff. Water temperature (deg C)",
        "TOC": "Total organic carbon (mg/L)", "TotN": "Total nitrogen (mg/L)",
        "TotP": "Total phosphorus (mg/L)", "NH4": "Ammonia (mg/L)"}
parameters = list(para.keys())
criteria_text = {"Chlfa": "50 ug/L", "OXY": "3 mg/L", "TOC": "10 mg/L", "TotN": "1 mg/L",
                 "TotP": "0.06 mg/L", "NH4": "0.5 mg/L"} 
criteria_value = {"Chlfa": 50, "OXY": 3, "TOC": 10, "TotN": 1, "TotP": 0.06, "NH4": 0.5}  

year = 2060

df2019_npv = data_import(str(year), 'NPV_day')
df2019_pv = data_import(str(year), 'FPV_day_PV1')

Temp_col = df2019_npv.columns[df2019_npv.columns.str.contains(pat = 'OXY')] 
selected_areas_col = [col for col in Temp_col if col.split("_", 1)[1] in area.keys()]

#Only temperature
df2019_npv = data_import(str(year), 'NPV_day')[selected_areas_col]
df2019_pv = data_import(str(year), 'FPV_day_PV1')[selected_areas_col]

df_Temp_diff_2019 = df2019_pv - df2019_npv
columns = df_Temp_diff_2019.columns

Temp_diff_exceedance_2019 = []

#2019 difference
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
fig, ax = plt.subplots()
fig.set_size_inches(4, 8.5)
df_Temp_diff_2019.boxplot(notch=True, widths = 0.4,
           # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
           # capprops=dict(color="steelblue"),
           whiskerprops=dict(color='gray',linewidth=1.5),
           flierprops=dict(color='gray',markersize=1.5),
           medianprops=dict(color="black"),
           # patch_artist = True,
           )
plt.xticks([1], list(area.values()), size = 18)
# plt.axhline(y = 3, color = 'orange', linewidth=2.5) 
plt.grid(visible=True,linestyle='-')
# plt.ylim([-0.25, 0.75])
ax.tick_params(direction="in")
plt.title('Year ' + str(year) + ", [PV-NPV]",size = 20)
plt.ylabel(para['OXY'],size = 20)
plt.savefig('OXY'+" Diff_BottomWater_PV1_"+str(year)+"_annual.png", bbox_inches='tight')
plt.close()

#2019 absolute
df = df2019_npv.merge(df2019_pv, how='outer', left_index=True, right_index=True, suffixes=('_NPV', '_PV'))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
fig, ax = plt.subplots()
fig.set_size_inches(8, 8.5)
axes = df.boxplot(notch=True, widths = 0.4,
           # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
           # capprops=dict(color="steelblue"),
           whiskerprops=dict(color='gray',linewidth=1.5),
           flierprops=dict(color='gray',markersize=1.5),
           medianprops=dict(color="black"),
           patch_artist = True,
           )
for k in range(len(df.columns)):
    if k<=0:
        axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='-',color='steelblue',linewidth=2.5)
        axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
        # axes.findobj(matplotlib.patches.Patch)[i].set_edgecolor('steelblue')
    else:
        axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='--',color='olivedrab',linewidth=2.5)
        axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
plt.grid(visible=True,linestyle='-')
plt.axhline(y = 3, color = 'orange', linewidth=2.5) 
xlim_original = ax.get_xlim()
plt.xlim(xlim_original)
ax.tick_params(direction="in")
# plt.text(1.5, maximum.max(), 'Without FPV', size = 25, color = 'steelblue')
# plt.text(5, maximum.max(), 'With FPV', size = 25, color = 'olivedrab')
plt.title('Year ' + str(year),size = 35)
plt.xticks([1, 2], list(area.values())*2, size = 20)
plt.ylabel(para['OXY'],size = 20)
plt.savefig('OXY'+"_BottomWater_PV1_"+str(year)+"_annual.png", bbox_inches='tight')
plt.close()
#%% Annual boxplot - absolute values (depth average balance areas)
directory = r"C:\Users\923627\OneDrive - Royal HaskoningDHV\Desktop\NgocNgoc\Work\LSR FPV EIA\Stratification"
os.chdir(directory)
area = {'LSR-W': 'Whole Reservoir',
        'FPV' : 'PV Area',
        'Outflow_200mBuffer' : 'Intake Area'}
para = {"Chlfa": "Chlorophyll-a (ug/L)", "OXY": "Dissolved oxygen (mg/L)",
        "PO4": "Phosphate (mg/L)",
        "SS": "Suspended solid (mg/L}", "Temp": "Water temperature (deg C)",
        "TOC": "Total organic carbon (mg/L)", "TotN": "Total nitrogen (mg/L)",
        "TotP": "Total phosphorus (mg/L)", "NH4": "Ammonia (mg/L)"}
parameters = list(para.keys())
criteria_text = {"Chlfa": "50 ug/L", "OXY": "3 mg/L", "TOC": "10 mg/L", "TotN": "1 mg/L",
                 "TotP": "0.06 mg/L", "NH4": "0.5 mg/L"} 
criteria_value = {"Chlfa": 50, "OXY": 3, "TOC": 10, "TotN": 1, "TotP": 0.06, "NH4": 0.5}  

year = 2060

df2019_npv = data_import(str(year), 'NPV_day')
df2019_pv = data_import(str(year), 'FPV_day')

#TOC-UlvaS
columns = df2019_npv.columns
TOC_col = df2019_npv.columns[df2019_npv.columns.str.contains(pat = 'TOC')] 
UlvaS_col = df2019_npv.columns[df2019_npv.columns.str.contains(pat = 'UlvaS')] 
selected_areas_col_TOC = [col for col in TOC_col if col.split("_", 1)[1] in area.keys()]
selected_areas_col_UlvaS = [col for col in UlvaS_col if col.split("_", 1)[1] in area.keys()]

df2019_npv_selected_TOC, df2019_pv_selected_TOC = df2019_npv[selected_areas_col_TOC], df2019_pv[selected_areas_col_TOC]
df2019_npv_selected_UlvaS, df2019_pv_selected_UlvaS = df2019_npv[selected_areas_col_UlvaS], df2019_pv[selected_areas_col_UlvaS]

df2019_npv_TOC = df2019_npv_selected_TOC - df2019_npv_selected_UlvaS.values
df2019_pv_TOC = df2019_pv_selected_TOC - df2019_pv_selected_UlvaS.values

df2019_npv[selected_areas_col_TOC] = df2019_npv_TOC
df2019_pv[selected_areas_col_TOC] = df2019_pv_TOC
df2019_npv = df2019_npv.drop(columns=df2019_npv_selected_UlvaS)
df2019_pv = df2019_pv.drop(columns=df2019_pv_selected_UlvaS)

#Plot
columns = df2019_npv.columns
selected_areas_col = [col for col in df2019_npv.columns if col.split("_", 1)[1] in area.keys()]
df2019_npv_selected, df2019_pv_selected = df2019_npv[selected_areas_col], df2019_pv[selected_areas_col]

plot_nocriteria(df2019_npv_selected,df2019_pv_selected, year, parameters, area)
#%% Annual boxplot - temperature absolute & diff (depth-average)
directory = r"C:\Users\923627\OneDrive - Royal HaskoningDHV\Desktop\NgocNgoc\Work\LSR FPV EIA\Stratification"
os.chdir(directory)
area = {'LSR-W_Bottom' : 'Whole Reservoir\nBottom water'}
# area = {'LSR-W' : 'Whole Reservoir'}

para = {"Chlfa": "Chlorophyll-a (ug/L)", "OXY": "Dissolved oxygen (mg/L)",
        "PO4": "Phosphate (mg/L)",
        "SS": "Suspended solid (mg/L}", "Temp": "Diff. Water temperature (deg C)",
        "TOC": "Total organic carbon (mg/L)", "TotN": "Total nitrogen (mg/L)",
        "TotP": "Total phosphorus (mg/L)", "NH4": "Ammonia (mg/L)"}
parameters = list(para.keys())
criteria_text = {"Chlfa": "50 ug/L", "OXY": "3 mg/L", "TOC": "10 mg/L", "TotN": "1 mg/L",
                 "TotP": "0.06 mg/L", "NH4": "0.5 mg/L"} 
criteria_value = {"Chlfa": 50, "OXY": 3, "TOC": 10, "TotN": 1, "TotP": 0.06, "NH4": 0.5}  

year = 2060

df2019_npv = data_import(str(year), 'NPV_day')
df2019_pv = data_import(str(year), 'FPV_day_PV1')

Temp_col = df2019_npv.columns[df2019_npv.columns.str.contains(pat = 'Temp')] 
selected_areas_col = [col for col in Temp_col if col.split("_", 1)[1] in area.keys()]

#Only temperature
df2019_npv = data_import(str(year), 'NPV_day')[selected_areas_col]
df2019_pv = data_import(str(year), 'FPV_day_PV1')[selected_areas_col]

df_Temp_diff_2019 = df2019_pv - df2019_npv
columns = df_Temp_diff_2019.columns

Temp_diff_exceedance_2019 = []

#2019 difference
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
fig, ax = plt.subplots()
fig.set_size_inches(4, 8.5)
df_Temp_diff_2019.boxplot(notch=True, widths = 0.4,
           # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
           # capprops=dict(color="steelblue"),
           whiskerprops=dict(color='gray',linewidth=1.5),
           flierprops=dict(color='gray',markersize=1.5),
           medianprops=dict(color="black"),
           # patch_artist = True,
           )
plt.xticks([1], list(area.values()), size = 18)
plt.axhline(y = 0.2, color = 'orange', linewidth=2.5) 
plt.grid(visible=True,linestyle='-')
plt.ylim([-1.0, 1.0])
ax.tick_params(direction="in")
plt.title('Year ' + str(year) + ", [PV-NPV]",size = 20)
plt.ylabel(para['Temp'],size = 20)
plt.savefig('Temp'+" Diff_BottomWater_PV1_"+str(year)+"_annual.png", bbox_inches='tight')
plt.close()

#2019 absolute
df = df2019_npv.merge(df2019_pv, how='outer', left_index=True, right_index=True, suffixes=('_NPV', '_PV'))
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
fig, ax = plt.subplots()
fig.set_size_inches(8, 8.5)
axes = df.boxplot(notch=True, widths = 0.4,
           # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
           # capprops=dict(color="steelblue"),
           whiskerprops=dict(color='gray',linewidth=1.5),
           flierprops=dict(color='gray',markersize=1.5),
           medianprops=dict(color="black"),
           patch_artist = True,
           )
for k in range(len(df.columns)):
    if k<=0:
        axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='-',color='steelblue',linewidth=2.5)
        axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
        # axes.findobj(matplotlib.patches.Patch)[i].set_edgecolor('steelblue')
    else:
        axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='--',color='olivedrab',linewidth=2.5)
        axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
plt.grid(visible=True,linestyle='-')
xlim_original = ax.get_xlim()
plt.xlim(xlim_original)
ax.tick_params(direction="in")
# plt.text(1.5, maximum.max(), 'Without FPV', size = 25, color = 'steelblue')
# plt.text(5, maximum.max(), 'With FPV', size = 25, color = 'olivedrab')
plt.title('Year ' + str(year),size = 35)
plt.xticks([1, 2], list(area.values())*2, size = 20)
plt.ylabel('Water temperature (deg C)',size = 20)       
plt.savefig('Temp'+"_BottomWater_PV1_"+str(year)+"_annual.png", bbox_inches='tight')
plt.close()
