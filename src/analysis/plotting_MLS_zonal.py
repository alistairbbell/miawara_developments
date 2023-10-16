#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:29:05 2023

Author: Alistair Bell

Contact: alistair.bell@unibe.ch
"""

#imports
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import os
import sys
sys.path.append('../../')
import matplotlib.dates as mdates
from utils import temps
import uuid
from smb.SMBConnection import SMBConnection
import logging
import traceback
from netCDF4 import Dataset
from logging.handlers import RotatingFileHandler
from matplotlib import animation
from matplotlib.animation import PillowWriter#%%
import matplotlib.colors as colors
import matplotlib.ticker as ticker

#%%
fig_file = "../../output/figs"

#%%2
my_xr_2022 = xr.open_dataset('/home/alistair/hunga_tonga/data/interim/MLS_Zonal_2022.nc')
my_xr_2023 = xr.open_dataset('/home/alistair/hunga_tonga/data/interim/MLS_Zonal_2023.nc')


#%%
my_xr_2022 = monthly_average = my_xr_2022.groupby('time.month').mean('time')
my_xr_2023 = monthly_average = my_xr_2023.groupby('time.month').mean('time')

#plot = my_xr.isel(Latitude_bins = 45).H2O.plot.contourf(x = 'time', y = 'nLevels', levels = 20, vmin = 1e-7, vmax = 100e-6)
# Extract pressure values corresponding to nLevels
pressure_values = np.array(my_xr_2023['Pressure'].values)
lat_values = np.array(my_xr_2023['Latitude'].values)

#%%
data_h2o = np.zeros([90,55,22])
data_prior = np.zeros([90,55,22])

for i in range(0,12):
    data_h2o[:,:,i] = np.nanmean(my_xr_2022.isel(month=slice(i, i+1)).H2O, axis = 0)
    data_prior[:,:,i] = np.nanmean(my_xr_2022.isel(month=slice(i, i+1))['H2O-APriori'], axis = 0)

for i in range(0,10):
    data_h2o[:,:,i+12] = np.nanmean(my_xr_2023.isel(month=slice(i, i+1)).H2O, axis = 0)
    data_prior[:,:,i+12] = np.nanmean(my_xr_2023.isel(month=slice(i, i+1))['H2O-APriori'], axis = 0)
    
# Create the plot
#%%
lat = np.array(my_xr_2022.Latitude[0,4:-4])
pressure = np.array(my_xr_2022.Pressure[0,45, :])
anom = data_h2o - data_prior

#%%set up plot
fig, ax = plt.subplots()
ax.set_xlim(-80, 80)
ax.set_ylim(1000, .0010)
ax.set_yscale('log')
ax.set_xlabel('Latitude (degrees)')
ax.set_ylabel('Pressure (hPa)')

min, max = -1000, 1000

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax

plotting_data = anom[4:-4, :, 0].T * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data) 

contour = ax.contourf(lat,pressure,plotting_data, levels=500, cmap=cmap, norm=colors.LogNorm(vmin=-min, vmax=max*10))
cb = plt.colorbar(contour)
#%%
class SymLogNorm(colors.Normalize):
    def __init__(self, linthresh, vmin=None, vmax=None, clip=False):
        self.linthresh = linthresh
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # Symmetric log normalization
        log_v = np.sign(value) * np.log1p(np.abs(value) / self.linthresh)
        return np.ma.masked_array(0.5 * (log_v / np.log1p(self.linthresh / self.vmin) + 1))

#%%set up plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(-80, 80)
ax.set_ylim(100, .0010)
ax.set_yscale('log')
ax.set_xlabel('Latitude (degrees)', fontsize = 20)
ax.set_ylabel('Pressure (hPa)', fontsize = 20)

min, max = -5, 5
levels = np.append(-np.logspace(1, -5, 70), np.logspace(-5, 1, 70))

# Initial contourf plot
cmap = plt.get_cmap('seismic')  # Use the 'viridis' colormap as a base
cmap.set_under('blue')  # Color for values < vmin
cmap.set_over('red')    # Color for values > vmax
ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=12)

plotting_data = anom[4:-4, :, 0].T * 1e6
plotting_data = np.where(plotting_data < min , min, plotting_data)
plotting_data = np.where(plotting_data > max , max, plotting_data) 
contour = ax.contourf(lat,pressure,plotting_data, levels=np.arange(-5,5,0.01), cmap=cmap)
cbar = plt.colorbar(contour )
cbar.set_label('Water Vapour Anomaly (PPMV)', fontsize = 20)
#cbar.ax.set_yscale('symlog')
cbar.ax.tick_params(labelsize=20)

#%%
def init():
    # Clear the previous contours
    for collection in contour.collections:
        collection.remove()
    return contour.collections,

def animate(i):
    # Plot the new contours for frame i
    plotting_data = anom[4:-4, :, i].T * 1e6
    plotting_data = np.where(plotting_data < min , min, plotting_data)
    plotting_data = np.where(plotting_data > max , max, plotting_data) 
    contour = ax.contourf(lat,pressure,plotting_data, levels=np.arange(-5,5,0.01), cmap=cmap)
    ax.set_title(f"Eruption + {i} months", fontsize = 20)
    plt.show()
    return contour.collections,

#%%
anim = animation.FuncAnimation(fig, animate,
                               frames=22, interval=1, blit=False)

savefile= os.path.join(fig_file, 'anim_hunga_tonga_lin.gif')
anim.save(savefile, writer=PillowWriter(fps=1))
