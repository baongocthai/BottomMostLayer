# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:48:06 2021

@author: liu
"""
import pandas as pd
from osgeo import ogr
from osgeo import gdal
import os
import math
import numpy as np
# import gdal
# import geopandas as gpd
# from shapely.geometry import Point, Polygon

#%% Check if is number
def isnumber(x):
    try:
        float(x)
        return True
    except:
        return False
    
#%% Function: Spatial met forcings
def rectangular_met_forcings(model_grid, sub_grid, pixel_width, pixel_height, output_name, quantity, parameters, minutes):
    bot_ds = ogr.Open(model_grid) # from D3D-FLOW input -- use Quickplot
    bot_layer = bot_ds.GetLayer()
    x_min, x_max, y_min, y_max = bot_layer.GetExtent()
    pixelWidth =  pixel_width #############Change if necessary - based on the grid
    pixelHeight = pixel_height
    #pixelWidth = pixelHeight = max([math.floor((x_max-x_min)/25),math.floor((y_max-y_min)/25)])
    cols = math.ceil((x_max - x_min) / pixelWidth)
    rows = math.ceil((y_max - y_min) / pixelHeight) 
    base_ds = gdal.GetDriverByName('GTiff').Create('base.tif', cols, rows, 1, gdal.GDT_Float32) 
    negativepixelHeight = pixelHeight*-1
    base_ds.SetGeoTransform((x_min, pixelWidth, 0, y_max, 0, negativepixelHeight))#origin at topleft
    band = base_ds.GetRasterBand(1)
    NoData_value = 0
    band.SetNoDataValue(NoData_value)
    band.FlushCache()
    # top_ds = ogr.Open("Model FPV layout_v2.shp") # shape file for FPV - only read geometry
    top_ds = ogr.Open(sub_grid) # shape file for FPV - only read geometry
    top_layer = top_ds.GetLayer()
    addvalue=float(1)
    gdal.RasterizeLayer(base_ds, [1], top_layer,burn_values=[addvalue])
    base_ds = None 
    k2=gdal.Open('base.tif').ReadAsArray() #PV
    k1=1-k2 #no PV
    ###########loop for different parameters##################
    for i,parameter_name in enumerate(quantity):
        #i=0        
        #name = "x_wind"
        #########################write dat file################
        filename = str(parameter_name+output_name)
        f= open(filename,"w")
        f.write("FileVersion      =    1.03"+"\n")
        f.write("filetype         =    meteo_on_equidistant_grid"+"\n")
        f.write("NODATA_value     =    -99.00"+"\n")
        f.write("n_cols           =    "+str(cols)+"\n")
        f.write("n_rows           =    "+str(rows)+'\n')
        f.write("grid_unit        =    m"+'\n')
        f.write("x_llcorner       =    "+str(int(x_min))+'\n')
        f.write("y_llcorner      =    "+str(int(y_min))+'\n')
        f.write("dx               =    "+str(pixelWidth)+'\n')
        f.write("dy               =    "+str(pixelHeight)+'\n')
        f.write("n_quantity       =    1"+'\n')
        f.write("quantity1        =    "+str(parameter_name)+'\n')
        f.write("unit1            =    "+str(unit[i]))
        
        #################insert time loop#####################
        for j in range(len(parameters)):
        #for j in [0,8759]:
        #for j in range(2):
            if pd.isnull(parameters[parameter_name][j]):
                continue
            else:
                f.write('\n'+"TIME = "+"{: >6}".format(str(j*minutes+0))+".0 minutes since 2018-12-01 00:00:00 +00:00"+"\n")#Change if necessary, to change the rolling time
                value1 = float(parameters[parameter_name][j])
                value2 = float(parameters_pv[parameter_name][j])
                k_sum = k1*value1 + k2*value2
                output = pd.DataFrame(k_sum).round(decimals=2)
                dfAsString = output.to_string(header=False, index=False, col_space=9, justify="right")
                f.write(dfAsString)
        print (parameter_name)
        f.close()

#%% convert wind speed and direction to wind xy
directory = r'C:\Users\trainee2\Desktop\LSR\MetInput_spatial'
os.chdir(directory)
df = pd.read_csv("GCM_wind.csv",parse_dates=[0],dayfirst=True)
df.columns=['datetime','speed','direction']
xy_wind = pd.DataFrame(columns=["datetime","x_wind","y_wind"])
for j in range(len(df)):
    if isnumber(df.speed[j]) == False or isnumber(df.direction[j]) == False:
        xwind = np.nan
        ywind = np.nan
    else:
        xwind = -float(df.speed[j])*math.sin(float(df.direction[j])*math.pi/180)
        ywind = -float(df.speed[j])*math.cos(float(df.direction[j])*math.pi/180)
    xy_wind = xy_wind.append({'datetime':df.datetime[j],'x_wind': xwind,'y_wind': ywind},ignore_index=True)
xy_wind.to_csv("GCM_xy_wind.csv",header=True,index=False)


#%% Rectangular grid (LSR) 
directory = r'C:\Users\923627\OneDrive - Royal HaskoningDHV\Desktop\NgocNgoc\Work\LSR FPV EIA\08_MetInput_spatial'
os.chdir(directory)

# 2019
quantity = ["x_wind","y_wind","relative_humidity","air_temperature","cloudiness","sw_radiation_flux","air_pressure"]
unit = ["m s-1", "m s-1","%","Celsius","%","W/m2","mbar"]
parameters_raw = pd.read_csv("parameters_LSR_2019_AllParameters_94SolarRadiation.csv",parse_dates = [0],dayfirst=True, encoding='unicode_escape')
parameters_raw.columns = ["datetime","x_wind","y_wind","relative_humidity","air_temperature","cloudiness","sw_radiation_flux","air_pressure"]
parameters=parameters_raw.copy()
parameters_pv = parameters.copy()
parameters_pv["x_wind"] = parameters["x_wind"]*0.1
parameters_pv["y_wind"] = parameters["y_wind"]*0.1
parameters_pv["relative_humidity"] = 12.68*np.exp(0.0216*parameters["relative_humidity"])
parameters_pv["air_temperature"] =  0.2074*(parameters["air_temperature"]**2)-10.099*parameters["air_temperature"]+148.25
parameters_pv["cloudiness"] = 100
parameters_pv["sw_radiation_flux"] = parameters["sw_radiation_flux"]*0.6
parameters_pv.to_csv("parameters_LSR_2019_AllParameters_94SolarRadiation_PV.csv",index=False)

# Gridded met forcings 2019 PV
model_grid = "HD grid.shp"
# sub_grid = "FPV_Aurecon_20231116.shp"
# sub_grid = "Model FPV layout_v4_dissolved.shp"
sub_grid = "Model_FPV_layout_v4_corridors_v2.shp"
pixel_width = 20
pixel_height = 20
# output_name = "_2019_LSR_PV_nocorridor_20x20.dat"
output_name = "_2019_LSR_PV_corridors_v2.dat"
quantity = quantity
parameters = parameters
rectangular_met_forcings(model_grid, sub_grid, pixel_width, pixel_height, output_name, quantity, parameters, 60)

## temperature 2019 offset for future years -- hourly
year = 2050
quantity = ["air_temperature"]
unit = ["Celsius","W/m2"]
parameters_raw = pd.read_csv("parameters_"+str(year)+"_hourly_temperature.csv",parse_dates = [0],dayfirst=True, encoding='unicode_escape')
parameters_raw.columns = ["datetime","air_temperature"]
parameters=parameters_raw.copy()
parameters_pv = parameters.copy()
parameters_pv["air_temperature"] = 0.2074*(parameters["air_temperature"]**2)-10.099*parameters["air_temperature"]+148.25
parameters_pv.to_csv("parameters_"+str(year)+"_hourly_temperature_PV.csv",index=False)

# Gridded met forcings future years
model_grid = "HD grid.shp"
# sub_grid = "FPV_Aurecon_20231116.shp"
# sub_grid = "Model FPV layout_v4_dissolved.shp"
sub_grid = "Model_FPV_layout_v4_corridors_v2.shp"
pixel_width = 20
pixel_height = 20
# output_name = "_"+str(year)+"_LSR_PV_nocorridor_2019OffSet_20x20.dat"
output_name = "_"+str(year)+"_LSR_PV_corridors_v2_2019OffSet_20x20.dat"
quantity = quantity
parameters = parameters
rectangular_met_forcings(model_grid, sub_grid, pixel_width, pixel_height, output_name, quantity, parameters, 60)

# =============================================================================
# # Future years GCM (not in use)
# ## Solar radiation & temperature -- hourly
# quantity = ["air_temperature","sw_radiation_flux"]
# unit = ["Celsius","W/m2"]
# parameters_raw = pd.read_csv("parameters_2060_hourly_temperature_solar.csv",parse_dates = [0],dayfirst=True, encoding='unicode_escape')
# parameters_raw.columns = ["datetime","air_temperature","sw_radiation_flux"]
# parameters=parameters_raw.copy()
# parameters_pv = parameters.copy()
# parameters_pv["air_temperature"] = 0.2074*(parameters["air_temperature"]**2)-10.099*parameters["air_temperature"]+148.25
# parameters_pv["sw_radiation_flux"] = parameters["sw_radiation_flux"]*0.6
# parameters_pv.to_csv("parameters_2060_hourly_temperature_solar_PV.csv",index=False)
# 
# # Gridded met forcings future years
# model_grid = "HD grid.shp"
# sub_grid = "Model FPV layout_v4_dissolved.shp"
# pixel_width = 20
# pixel_height = 20
# output_name = "_2060_LSR_PV_nocorridor_20x20.dat"
# quantity = quantity
# parameters = parameters
# rectangular_met_forcings(model_grid, sub_grid, pixel_width, pixel_height, output_name, quantity, parameters, 60)
# 
# ## Other met forcings -- daily average
# quantity = ["x_wind","y_wind","cloudiness","relative_humidity"]
# unit = ["m s-1", "m s-1","%","%"]
# parameters_raw = pd.read_csv("parameters_2040_other_met_parameters.csv",parse_dates = [0],dayfirst=True, encoding='unicode_escape')
# parameters_raw.columns = ["datetime","x_wind","y_wind","cloudiness","relative_humidity"]
# parameters=parameters_raw.copy()
# parameters_pv = parameters.copy()
# parameters_pv["x_wind"] = parameters["x_wind"]*0.1
# parameters_pv["y_wind"] = parameters["y_wind"]*0.1
# parameters_pv["relative_humidity"] = 12.68*np.exp(0.0216*parameters["relative_humidity"])
# parameters_pv["cloudiness"] = 100
# parameters_pv.to_csv("parameters_2040_other_met_parameters_PV.csv",index=False)
# 
# # Gridded met forcings future years
# model_grid = "HD grid.shp"
# sub_grid = "Model FPV layout_v4_dissolved.shp"
# pixel_width = 20
# pixel_height = 20
# output_name = "_2040_LSR_PV_nocorridor_20x20.dat"
# quantity = quantity
# parameters = parameters
# rectangular_met_forcings(model_grid, sub_grid, pixel_width, pixel_height, output_name, quantity, parameters, 1440)
# =============================================================================



#%% Rectangular grid (LSR) - solar radiation PV = 60% solar radiation NPV
directory = r'E:\MetInput_spatial'
os.chdir(directory)

# 2019
quantity = ["sw_radiation_flux"]
unit = ["W/m2"]
parameters_raw = pd.read_csv("parameters_LSR_2019_94SolarRadiation.csv",parse_dates = [0],dayfirst=True, encoding='unicode_escape')
parameters_raw.columns = ["datetime","sw_radiation_flux"]
parameters=parameters_raw.copy()
parameters_pv = parameters.copy()
parameters_pv["sw_radiation_flux"] = parameters["sw_radiation_flux"]*0.6

parameters_pv.to_csv("parameters_LSR_2019_94SolarRadiation_PV_40NPV.csv",index=False)

# Gridded met forcings 2019 NPV
model_grid = "HD grid.shp"
sub_grid = "FPV_Aurecon_20231116.shp"
pixel_width = 20
pixel_height = 20
output_name = "_2019_LSR_PV_60NPV.dat"
quantity = quantity
parameters = parameters
rectangular_met_forcings(model_grid, sub_grid, pixel_width, pixel_height, output_name, quantity, parameters, 60)
#%% Rectangular grid (LSR) - wind speed PV = 40% wind speed NPV
directory = r'E:\MetInput_spatial'
os.chdir(directory)

# 2019
quantity = ["x_wind","y_wind"]
unit = ["m s-1", "m s-1"]
parameters_raw = pd.read_csv("parameters_LSR_2019_wind.csv",parse_dates = [0],dayfirst=True, encoding='unicode_escape')
parameters_raw.columns = ["datetime","x_wind","y_wind"]
parameters=parameters_raw.copy()
parameters_pv = parameters.copy()
parameters_pv["x_wind"] = parameters["x_wind"]*0.4
parameters_pv["y_wind"] = parameters["y_wind"]*0.4

parameters_pv.to_csv("parameters_LSR_2019_wind_PV_40NPV.csv",index=False)

# Gridded met forcings 2019 NPV
model_grid = "HD grid.shp"
sub_grid = "Model FPV layout_v4_dissolved.shp"
pixel_width = 20
pixel_height = 20
output_name = "_2019_LSR_PV_40NPV.dat"
quantity = quantity
parameters = parameters
rectangular_met_forcings(model_grid, sub_grid, pixel_width, pixel_height, output_name, quantity, parameters, 60)
