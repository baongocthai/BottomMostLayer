# -*- coding: utf-8 -*-
"""
Box Plot for Pandan FPV EIA
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
from functools import reduce
import matplotlib.patches as mpatches
import matplotlib

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
#%% Plot - no criteria line
def plot_nocriteria(npv, pv, year):
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
        plt.title('Year ' + str(year),size = 35)
        plt.xticks([1, 2, 3, 4, 5, 6], list(area.values())*2, size = 20)
        plt.ylabel(para[parameters[i]],size = 20)       
        plt.savefig(parameters[i]+"_"+str(year)+"_annual.png", bbox_inches='tight')
        print(parameters[i])
        plt.close()

#%% Plot Dt NPV-PV
def plot_temp_diff(temp_diff, year):   
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
     
#%% Annual boxplot - absolute values - comparison for 3 balance areas & 2 years
directory = r"C:\Users\baongoc.thai\OneDrive - Hydroinformatics Institute Pte Ltd\Desktop\Work\5. Pandan HD\Pandan"
os.chdir(directory)
area = {'Whole-Reservoir_segm': 'Whole Reservoir',
        'PV_withCorridor' : 'PV Area',
        'CCKWW_200m' : 'Intake Area'}
para = {"Chlfa": "Chlorophyll-a (ug/L)", "OXY": "Dissolved oxygen (mg/L)",
        "PO4": "Phosphate (mg/L)",
        "SS": "Suspended solid (mg/L}", "Temp": "Water temperature (deg C)",
        "TOC": "Total organic carbon (mg/L)", "TotN": "Total nitrogen (mg/L)",
        "TotP": "Total phosphorus (mg/L)", "NH4": "Ammonia (mg/L)"}
parameters = list(para.keys())
criteria_text = {"Chlfa": "50 ug/L", "OXY": "3 mg/L", "TOC": "10 mg/L", "TotN": "1 mg/L",
                 "TotP": "0.06 mg/L", "NH4": "0.5 mg/L"} 
criteria_value = {"Chlfa": 50, "OXY": 3, "TOC": 10, "TotN": 1, "TotP": 0.06, "NH4": 0.5}  

df2019_npv = data_import(2019, 'NPV_day')
df2019_pv = data_import(2019, 'PV_day')
df2060_npv = data_import(2060, 'NPV_day')
df2060_pv = data_import(2060, 'PV_day')

columns = df2019_npv.columns
selected_areas_col = [col for col in df2019_npv.columns if col.split("_", 2)[2] in area.keys()]

df2019_npv_selected, df2019_pv_selected = df2019_npv[selected_areas_col], df2019_pv[selected_areas_col]
df2060_npv_selected, df2060_pv_selected = df2060_npv[selected_areas_col], df2060_pv[selected_areas_col]

plot_nocriteria(df2019_npv_selected,df2019_pv_selected, 2019)
plot_nocriteria(df2060_npv_selected,df2060_pv_selected, 2060)

#2019
# =============================================================================
# for i in range(len(parameters)):
#     df_temp1 = df2019_npv_selected.filter(regex=parameters[i])
#     df_temp2 = df2019_pv_selected.filter(regex=parameters[i])
#     df = df_temp1.merge(df_temp2, how='outer', left_index=True, right_index=True, suffixes=('_NPV', '_PV'))
#     iqr = np.percentile(df, 75) - np.percentile(df, 25)
#     maximum = df.max()+0.5*iqr
#     minimum = df.min()-0.5*iqr
#     
#     #Plot
#     plt.rc('xtick', labelsize=18)
#     plt.rc('ytick', labelsize=18)
#     fig, ax = plt.subplots()
#     fig.set_size_inches(18, 8)
#     axes = df.boxplot(notch=True, widths = 0.4,
#                # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
#                # capprops=dict(color="steelblue"),
#                whiskerprops=dict(color='gray',linewidth=1.5),
#                flierprops=dict(color='gray',markersize=1.5),
#                medianprops=dict(color="black"),
#                patch_artist = True,
#                )
#     for k in range(len(df.columns)):
#         if k<=2:
#             axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='-',color='steelblue',linewidth=2.5)
#             axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
#             # axes.findobj(matplotlib.patches.Patch)[i].set_edgecolor('steelblue')
#         else:
#             axes.findobj(matplotlib.patches.Patch)[k].set(linestyle='--',color='olivedrab',linewidth=2.5)
#             axes.findobj(matplotlib.patches.Patch)[k].set_facecolor('None')
#             # axes.findobj(matplotlib.patches.Patch)[i].set_edgecolor('olivedrab')
# 
#     
#     if parameters[i] in criteria_text:
#         ax.axhline(y=criteria_value[parameters[i]], color='orange', linewidth=3, label = criteria_text[parameters[i]])
#         handles, labels = ax.get_legend_handles_labels()
#         patch1 = mpatches.Patch(edgecolor='steelblue', facecolor="white", label='NPV')
#         patch2 = mpatches.Patch(edgecolor='olivedrab', facecolor="white", linestyle='--', label='PV')
#         handles.append(patch1)
#         handles.append(patch2)
#         handles_arr = [handles[i] for i in [1,2,0]]
#     else:
#         patch1 = mpatches.Patch(edgecolor='steelblue', facecolor="white", label='NPV')
#         patch2 = mpatches.Patch(edgecolor='olivedrab', facecolor="white", linestyle='--', label='PV')
#         handles_arr = [patch1, patch2]
#     try:
#         if parameters[i] == 'Temp':
#             plt.ylim([minimum.min(), maximum.max()])
#         elif parameters[i] == 'OXY':
#             plt.ylim([0, maximum.max()])
#         elif maximum.max() < criteria_value[parameters[i]]:
#             plt.ylim([minimum.min(), criteria_value[parameters[i]]])
#         elif minimum.min() > criteria_value[parameters[i]]:
#             plt.ylim([criteria_value[parameters[i]], maximum.max()])
#     except KeyError:
#         plt.ylim([minimum.min(), maximum.max()])    
#     
#     if parameters[i] == 'OXY':
#         ax.legend(handles = handles_arr, loc='lower left',fontsize=20)
#     else:
#         ax.legend(handles = handles_arr, loc='upper left',fontsize=20)   
#     
#     plt.grid(visible=True,linestyle='-')
#     ax.axvline(x=3.5, color='black', linestyle=':', linewidth=2)
#     xlim_original = ax.get_xlim()
#     plt.xlim(xlim_original)
#     ax.tick_params(direction="in")
#     plt.text(1.5, maximum.max()+maximum.max()*0.01, 'NPV', size = 25, color = 'steelblue')
#     plt.text(5, maximum.max()+maximum.max()*0.01, 'PV', size = 25, color = 'olivedrab')
#     # plt.text(1.5, maximum.max(), 'NPV'+'\n'+'PV', size = 25, color = 'steelblue')
#     # plt.text(5, maximum.max(), 'PV', size = 25, color = 'olivedrab')
#     plt.title('Year 2019',size = 35)
#     plt.xticks([1, 2, 3, 4, 5, 6], list(area.values())*2, size = 18)
#     plt.ylabel(para[parameters[i]],size = 18)       
#     plt.savefig(parameters[i]+"_2019_annual.png", bbox_inches='tight')
#     print(parameters[i])
#     plt.close()
# =============================================================================
#%% Annual boxplot - absolute values
directory = r"C:\Users\baongoc.thai\OneDrive - Hydroinformatics Institute Pte Ltd\Desktop\Work\5. Pandan HD\Pandan"
os.chdir(directory)
area = {'Whole-Reservoir_segm': 'Whole Reservoir - Depth average',
        'PV_2blocks' : 'Balance Area PV - Depth average',
        'PV_BlockLeft' : 'PV block left - Depth average',
        'PV_BlockRight' : 'PV block right - Depth average',
        'PV_BlockRight_Surf' : 'PV block right - Surface',
        'PV_BlockLeft_Surf' : 'PV block left - Surface',
        'PV_2blocks_Surf' : 'Balance Area PV - Surface',
        'CCKWW': 'Balance Area CCKWW Intake - Depth average',
        'Bottom-water_4-5m' : '4.5m depth water column',
        'Bottom-water_2m' : '2m depth water column'}
para = {"Chlfa": "Chlorophyll-a (ug/L)", "OXY": "Dissolved oxygen (mg/L)",
        "PO4": "Phosphate (mg/L)",
        "SS": "Suspended solid (mg/L}", "Temp": "Temperature (deg C)",
        "TOC": "Total organic carbon (mg/L)", "TotN": "Total nitrogen (mg/L)",
        "TotP": "Total phosphorus (mg/L)", "NH4": "Ammonia (mg/L)"}
criteria_text = {"Chlfa": "50 ug/L", "OXY": "3 mg/L", "TOC": "10 mg/L", "TotN": "1 mg/L",
                 "TotP": "0.06 mg/L", "NH4": "0.5 mg/L"} 
criteria_value = {"Chlfa": 50, "OXY": 3, "TOC": 10, "TotN": 1, "TotP": 0.06, "NH4": 0.5}  

df2019_npv = data_import(2019, 'NPV_day')
df2019_pv = data_import(2019, 'PV_day')
df2060_npv = data_import(2060, 'NPV_day')
df2060_pv = data_import(2060, 'PV_day')

columns = df2019_npv.columns

for i in range(len(columns)):
    df = [df2019_npv[columns[i]], df2019_pv[columns[i]], df2060_npv[columns[i]], df2060_pv[columns[i]]]
    df = pd.DataFrame(df).transpose()
    df.columns = ['NPV-2019','PV-2019','NPV-2060','PV-2060']
    iqr = np.percentile(df, 75) - np.percentile(df, 25)
    maximum = df.max()+0.5*iqr
    minimum = df.min()-0.5*iqr
    parameter = columns[i].split("_", 2)[0]
    layer = columns[i].split("_", 2)[2]
    
    #Plot
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    fig, ax = plt.subplots()
    for p,s in enumerate(df.columns):
        if p%2 == 0:
            ls = '-'
            if p==0:
                bp1 = ax.boxplot(df.iloc[:,p].dropna(), positions=[p+1], notch=True,
                labels = ["Base"], widths = 0.4,
                boxprops=dict(color="steelblue",linestyle=ls, linewidth=2.5),
                capprops=dict(color="steelblue"),
                whiskerprops=dict(color="steelblue", linestyle=ls, linewidth=2.5),
                flierprops=dict(color="steelblue", markeredgecolor="steelblue",markersize=3),
                medianprops=dict(color="steelblue"),zorder=p
                )
            else:
                bp3 = ax.boxplot(df.iloc[:,p].dropna(), positions=[p+1], notch=True,
                labels = ["NPV"], widths = 0.4,
                boxprops=dict(color="steelblue", linestyle=ls, linewidth=2.5),
                capprops=dict(color="steelblue"),
                whiskerprops=dict(color="steelblue", linestyle=ls, linewidth=2.5),
                flierprops=dict(color="steelblue", markeredgecolor="steelblue",markersize=3),
                medianprops=dict(color="steelblue"),zorder=p
                )
        else:
            ls = '--'
            if p==1:
                bp2 = ax.boxplot(df.iloc[:,p].dropna(), positions=[p+1], notch=True,
                labels = ["PV"], widths = 0.4, 
                boxprops=dict(color="olivedrab", linestyle=ls, linewidth=2.5),
                capprops=dict(color="olivedrab"),
                whiskerprops=dict(color="olivedrab", linestyle=ls, linewidth=2.5),
                flierprops=dict(color="olivedrab", markeredgecolor="olivedrab",markersize=3),
                medianprops=dict(color="olivedrab"),zorder=p
                )
            else:
                bp4 = ax.boxplot(df.iloc[:,p].dropna(), positions=[p+1], notch=True,
                labels = ["PV"], widths = 0.4,
                boxprops=dict(color="olivedrab", linestyle=ls, linewidth=2.5),
                capprops=dict(color="olivedrab"),
                whiskerprops=dict(color="olivedrab", linestyle=ls, linewidth=2.5),
                flierprops=dict(color="olivedrab", markeredgecolor="olivedrab",markersize=3),
                medianprops=dict(color="olivedrab"),zorder=p
                )
        xlim_original = ax.get_xlim()
    
    if parameter in criteria_text:
        criteria = np.empty(10)
        criteria.fill(criteria_value[parameter])
        l1 = ax.plot(criteria,color = "orange", linewidth=3, 
                     label = criteria_text[parameter], zorder=p+1)
        handles, labels = ax.get_legend_handles_labels()
        patch1 = mpatches.Patch(edgecolor='steelblue', facecolor="white", label='NPV')
        patch2 = mpatches.Patch(edgecolor='olivedrab', facecolor="white", linestyle='--', label='PV')
        handles.append(patch1)
        handles.append(patch2)
        handles_arr = [handles[i] for i in [1,2,0]]
    else:
        patch1 = mpatches.Patch(edgecolor='steelblue', facecolor="white", label='NPV')
        patch2 = mpatches.Patch(edgecolor='olivedrab', facecolor="white", linestyle='--', label='PV')
        handles_arr = [patch1, patch2]
    ax.legend(handles = handles_arr, loc='upper left',fontsize=20)
    
    plt.grid(visible=True,linestyle='-')
    fig.text(0.215, 0.085, "2019", va='center', rotation='horizontal',size=15)
    fig.text(0.41, 0.085, "2019", va='center', rotation='horizontal',size=15)
    fig.text(0.60, 0.085, "2060", va='center', rotation='horizontal',size=15)
    fig.text(0.795, 0.085, "2060", va='center', rotation='horizontal',size=15)
    try:
        if parameter == 'Temp':
            plt.ylim([minimum.min(), maximum.max()])    
        elif maximum.max() < criteria_value[parameter]:
            plt.ylim([minimum.min(), criteria_value[parameter]])
        elif minimum.min() > criteria_value[parameter]:
            plt.ylim([criteria_value[parameter], maximum.max()])
    except KeyError:
        plt.ylim([minimum.min(), maximum.max()])    
    plt.xlim(xlim_original)
    ax.tick_params(direction="in")
    plt.title(area[layer],size = 20)
    plt.ylabel(para[parameter],size = 20)
    
    fig.set_size_inches(15, 10)
    plt.savefig(parameter+"_"+layer+"_annual.png", bbox_inches='tight')
    plt.close()

#%% Annual boxplot - temperature diff
directory = r"C:\Users\baongoc.thai\OneDrive - Hydroinformatics Institute Pte Ltd\Desktop\Work\5. Pandan HD\Pandan"
os.chdir(directory)
area = {'Whole-Reservoir_segm': 'Whole Reservoir',
        'PV_withCorridor' : 'PV Area',
        'CCKWW_200m' : 'Intake Area'}
para = {"Chlfa": "Chlorophyll-a (ug/L)", "OXY": "Dissolved oxygen (mg/L)",
        "PO4": "Phosphate (mg/L)",
        "SS": "Suspended solid (mg/L}", "Temp": "Diff. Water temperature (deg C)",
        "TOC": "Total organic carbon (mg/L)", "TotN": "Total nitrogen (mg/L)",
        "TotP": "Total phosphorus (mg/L)", "NH4": "Ammonia (mg/L)"}
parameters = list(para.keys())
criteria_text = {"Chlfa": "50 ug/L", "OXY": "3 mg/L", "TOC": "10 mg/L", "TotN": "1 mg/L",
                 "TotP": "0.06 mg/L", "NH4": "0.5 mg/L"} 
criteria_value = {"Chlfa": 50, "OXY": 3, "TOC": 10, "TotN": 1, "TotP": 0.06, "NH4": 0.5}  

df2019_npv = data_import(2019, 'NPV_day')
df2019_pv = data_import(2019, 'PV_day')
df2060_npv = data_import(2060, 'NPV_day')
df2060_pv = data_import(2060, 'PV_day')

Temp_col = df2019_npv.columns[df2019_npv.columns.str.contains(pat = 'Temp')] 
selected_areas_col = [col for col in Temp_col if col.split("_", 2)[2] in area.keys()]

#Only temperature
df2019_npv = data_import(2019, 'NPV_day')[selected_areas_col]
df2019_pv = data_import(2019, 'PV_day')[selected_areas_col]
df2060_npv = data_import(2060, 'NPV_day')[selected_areas_col]
df2060_pv = data_import(2060, 'PV_day')[selected_areas_col]

df_Temp_diff_2019 = df2019_pv - df2019_npv
df_Temp_diff_2060 = df2060_pv - df2060_npv

columns = df_Temp_diff_2019.columns

Temp_diff_exceedance_2019 = []
Temp_diff_exceedance_2060 = []

plot_temp_diff(df_Temp_diff_2019,2019)
plot_temp_diff(df_Temp_diff_2060,2060)

#Plot
# =============================================================================
# plt.rc('xtick', labelsize=18)
# plt.rc('ytick', labelsize=18)
# fig, ax = plt.subplots()
# fig.set_size_inches(18, 8)
# axes = df_Temp_diff_2019.boxplot(notch=True, widths = 0.4,
#            # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
#            # capprops=dict(color="steelblue"),
#            whiskerprops=dict(color='gray',linewidth=1.5),
#            flierprops=dict(color='gray',markersize=1.5),
#            medianprops=dict(color="black"),
#            # patch_artist = True,
#            )
# plt.xticks([1, 2, 3], list(area.values()), size = 18)
# plt.axhline(y = 0.2, color = 'orange', linewidth=2.5) 
# plt.grid(visible=True,linestyle='-')
# plt.ylim([-0.25, 0.75])
# ax.tick_params(direction="in")
# plt.title('Year 2019' + ", [PV-NPV]",size = 20)
# plt.ylabel(para[parameter],size = 20)
# plt.savefig(parameter+" Diff_"+layer+"_annual.png", bbox_inches='tight')
# plt.close()
# =============================================================================

# pd.DataFrame(Temp_diff_exceedance_2019,columns).to_csv('Temperature Difference_exceedance2019.csv')
# pd.DataFrame(Temp_diff_exceedance_2060,columns).to_csv('Temperature Difference_exceedance2060.csv')

#%% Annual boxplot - temperature diff (bottom-water, temporary)
directory = r"C:\Users\baongoc.thai\OneDrive - Hydroinformatics Institute Pte Ltd\Desktop\Work\5. Pandan HD\Pandan"
os.chdir(directory)
area = {'Bottom-water_2m' : 'Whole Reservoir\nBottom water'}
para = {"Chlfa": "Chlorophyll-a (ug/L)", "OXY": "Dissolved oxygen (mg/L)",
        "PO4": "Phosphate (mg/L)",
        "SS": "Suspended solid (mg/L}", "Temp": "Diff. Water temperature (deg C)",
        "TOC": "Total organic carbon (mg/L)", "TotN": "Total nitrogen (mg/L)",
        "TotP": "Total phosphorus (mg/L)", "NH4": "Ammonia (mg/L)"}
parameters = list(para.keys())
criteria_text = {"Chlfa": "50 ug/L", "OXY": "3 mg/L", "TOC": "10 mg/L", "TotN": "1 mg/L",
                 "TotP": "0.06 mg/L", "NH4": "0.5 mg/L"} 
criteria_value = {"Chlfa": 50, "OXY": 3, "TOC": 10, "TotN": 1, "TotP": 0.06, "NH4": 0.5}  

df2019_npv = data_import(2019, 'NPV_day')
df2019_pv = data_import(2019, 'PV_day')
df2060_npv = data_import(2060, 'NPV_day')
df2060_pv = data_import(2060, 'PV_day')

Temp_col = df2019_npv.columns[df2019_npv.columns.str.contains(pat = 'Temp')] 
selected_areas_col = [col for col in Temp_col if col.split("_", 2)[2] in area.keys()]

#Only temperature
df2019_npv = data_import(2019, 'NPV_day')[selected_areas_col]
df2019_pv = data_import(2019, 'PV_day')[selected_areas_col]
df2060_npv = data_import(2060, 'NPV_day')[selected_areas_col]
df2060_pv = data_import(2060, 'PV_day')[selected_areas_col]

df_Temp_diff_2019 = df2019_pv - df2019_npv
df_Temp_diff_2060 = df2060_pv - df2060_npv

columns = df_Temp_diff_2019.columns

Temp_diff_exceedance_2019 = []
Temp_diff_exceedance_2060 = []

#2019
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
fig, ax = plt.subplots()
fig.set_size_inches(9, 8)
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
plt.title('Year ' + str(2019) + ", [PV-NPV]",size = 20)
plt.ylabel(para['Temp'],size = 20)
plt.savefig('Temp'+" Diff_BottomWWater_"+str(2019)+"_annual.png", bbox_inches='tight')
plt.close()

#2060
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
fig, ax = plt.subplots()
fig.set_size_inches(4, 8.5)
df_Temp_diff_2060.boxplot(notch=True, widths = 0.4,
           # boxprops=dict(color="steelblue",linestyle='-', linewidth=2.5),
           # capprops=dict(color="steelblue"),
           whiskerprops=dict(color='gray',linewidth=1.5),
           flierprops=dict(color='gray',markersize=1.5),
           medianprops=dict(color="black"),
           # patch_artist = True,
           )
plt.xticks([1], list(area.values()), size = 20)
plt.axhline(y = 0.2, color = 'orange', linewidth=2.5) 
plt.grid(visible=True,linestyle='-')
plt.ylim([-0.25, 0.75])
ax.tick_params(direction="in")
# plt.title('Year ' + str(2060) + ", [PV-NPV]",size = 20)
plt.suptitle('Year ' + str(2060),size = 35)
plt.title("[PV-NPV]",size = 20)
plt.ylabel(para['Temp'],size = 20)
plt.savefig('Temp'+" Diff_BottomWater_"+str(2060)+"_annual.png", bbox_inches='tight')
plt.close()
#%% Exceedance table
directory = "C:\\Users\\liu.H2I\\Desktop\\Kranji\\Scenario1\\WAQ_results\\"
os.chdir(directory)

df1=pd.read_csv("waq_final_2050base.csv",parse_dates = [0],dayfirst=True,skiprows=range(0,5))
col = df1.columns.to_list()
col_corrected = ["Date"] + col[1:]
df1.columns = col_corrected
df2 = pd.read_csv("waq_final_2019base.csv",parse_dates = [0],dayfirst=True,skiprows=range(0,5))
col = df2.columns.to_list()
col_corrected = ["Date"] + col[1:]
df2.columns = col_corrected
df3=pd.read_csv("waq_final_2050pv.csv",parse_dates = [0],dayfirst=True,skiprows=range(0,5))
col = df3.columns.to_list()
col_corrected = ["Date"] + col[1:]
df3.columns = col_corrected
df4 = pd.read_csv("waq_final_2019pv.csv",parse_dates = [0],dayfirst=True,skiprows=range(0,5))
col = df4.columns.to_list()
col_corrected = ["Date"] + col[1:]
df4.columns = col_corrected
df5=pd.read_csv("waq_final_2030base.csv",parse_dates = [0],dayfirst=True,skiprows=range(0,5))
col = df5.columns.to_list()
col_corrected = ["Date"] + col[1:]
df5.columns = col_corrected
df6 = pd.read_csv("waq_final_2030pv.csv",parse_dates = [0],dayfirst=True,skiprows=range(0,5))
col = df6.columns.to_list()
col_corrected = ["Date"] + col[1:]
df6.columns = col_corrected
df7=pd.read_csv("waq_final_2040base.csv",parse_dates = [0],dayfirst=True,skiprows=range(0,5))
col = df7.columns.to_list()
col_corrected = ["Date"] + col[1:]
df7.columns = col_corrected
df8 = pd.read_csv("waq_final_2040pv.csv",parse_dates = [0],dayfirst=True,skiprows=range(0,5))
col = df8.columns.to_list()
col_corrected = ["Date"] + col[1:]
df8.columns = col_corrected

for i,column in enumerate(df2): 
    if i == 0:
        continue
    parameter = column[:column.find("_")]
    regex=re.compile(r"^[^_]*_[^_]*_(.*)$")
    layer = re.sub(regex,r"\1",column)
    if layer !="Total System":
        continue
    if parameter != "OXY":
        continue
    df2019 = pd.DataFrame()
    for j, column2 in enumerate(df4):
        parameter2 = column2[:column2.find("_")]
        layer2 = re.sub(regex,r"\1",column2)
        if (parameter2 == parameter) & (layer2 == layer):
            df2019=pd.concat([df2[['Date']],df2[[column]],df4[[column2]]],axis=1)
            df2019.columns=['time','Base-2019',"PV-2019"]
            data1 = df2019[['Base-2019',"PV-2019"]].dropna()
            iqr1 = np.percentile(data1, 75) - np.percentile(data1, 25)
            max1 = data1.max()+0.5*iqr1
            min1 = data1.min()-0.5*iqr1
            break
    df2050 = pd.DataFrame()    
    done = 0
    for m, column3 in enumerate(df1):
        parameter3 = column3[:column3.find("_")]
        layer3 = re.sub(regex,r"\1",column3)
        if (parameter3 == parameter) & (layer3 == layer):
            for n, column4 in enumerate(df3):
                parameter4 = column4[:column4.find("_")]
                layer4 = re.sub(regex,r"\1",column4)
                if (parameter4 == parameter) & (layer4 == layer):
                    df2050=pd.concat([df1[['Date']],df1[[column3]],df3[[column4]]],axis=1)
                    df2050.columns=['time','Base-2050',"PV-2050"]
                    data2 = df2050[['Base-2050',"PV-2050"]].dropna()
                    iqr2 = np.percentile(data2, 75) - np.percentile(data2, 25)
                    max2 = data2.max()+0.5*iqr2
                    min2 = data2.min()-0.5*iqr2 
                    max_ = np.maximum(max1.max(),max2.max())
                    min_ = np.minimum(min1.min(),min2.min())
                    done = 1
                    break
        if done == 1:
            break
        
    df2030 = pd.DataFrame()    
    done = 0
    for m, column5 in enumerate(df5):
        parameter5 = column5[:column5.find("_")]
        layer5 = re.sub(regex,r"\1",column5)
        if (parameter5 == parameter) & (layer5 == layer):
            for n, column6 in enumerate(df6):
                parameter6 = column6[:column6.find("_")]
                layer6 = re.sub(regex,r"\1",column6)
                if (parameter6 == parameter) & (layer6 == layer):
                    df2030=pd.concat([df5[['Date']],df5[[column5]],df6[[column6]]],axis=1)
                    df2030.columns=['time','Base-2030',"PV-2030"]
                    data3 = df2030[['Base-2030',"PV-2030"]].dropna()
                    iqr3 = np.percentile(data3, 75) - np.percentile(data3, 25)
                    max3 = data3.max()+0.5*iqr2
                    min3 = data3.min()-0.5*iqr2  
                    max_ = np.maximum(max_,max3.max())
                    min_ = np.minimum(min_,min3.min())
                    done = 1
                    break
        if done == 1:
            break
    
    df2040 = pd.DataFrame()    
    done = 0
    for m, column7 in enumerate(df7):
        parameter7 = column7[:column7.find("_")]
        layer7 = re.sub(regex,r"\1",column7)
        if (parameter7 == parameter) & (layer7 == layer):
            for n, column8 in enumerate(df8):
                parameter8 = column8[:column8.find("_")]
                layer8 = re.sub(regex,r"\1",column8)
                if (parameter8 == parameter) & (layer8 == layer):
                    df2040=pd.concat([df7[['Date']],df7[[column7]],df8[[column8]]],axis=1)
                    df2040.columns=['time','Base-2040',"PV-2040"]
                    data4 = df2040[['Base-2040',"PV-2040"]].dropna()
                    iqr4 = np.percentile(data4, 75) - np.percentile(data4, 25)
                    max4 = data4.max()+0.5*iqr2
                    min4 = data4.min()-0.5*iqr2   
                    max_ = np.maximum(max_,max4.max())
                    min_ = np.minimum(min_,min4.min())
                    done = 1
                    break
        if done == 1:
            break
    data_frames = [df2019, df2030, df2040, df2050]    
    df = reduce(lambda left,right: pd.merge(left,right,on=['time'],how='outer'), data_frames)
    df.set_index("time",inplace=True)
temp_exceedance = pd.DataFrame(columns= ["Scenario","25th percentile","50th percentile",
                                         "75th percentile","99th percentile","exceedence"])
#year0=0
for p, column in enumerate(df):
    #year1 = column[column.find("-")+1:]
    #if year1 == year0:
    #    continue
    #year0 = year1
    #temp_diff = df["PV-"+str(year1)] - df["Base-"+str(year1)]
    temp_diff = df[column]
    per25 = np.percentile(temp_diff,25)
    per50 = np.percentile(temp_diff,50)
    per75 = np.percentile(temp_diff,75)
    per99 = np.percentile(temp_diff,99)
    exceedence = len(temp_diff[temp_diff<3])/len(temp_diff)
    temp_exceedance = temp_exceedance.append({"Scenario":column, "25th percentile": per25, 
                                              "50th percentile": per50, "75th percentile": per75,
                                              "99th percentile": per99, "exceedence": exceedence},
                                              ignore_index=True)
temp_exceedance.to_csv("exceedance_DO.csv",index=False)    
