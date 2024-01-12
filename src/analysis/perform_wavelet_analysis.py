#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:43:04 2024

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
import uuid
from smb.SMBConnection import SMBConnection
import logging
import traceback
from netCDF4 import Dataset
from logging.handlers import RotatingFileHandler
from matplotlib import animation
from matplotlib.animation import PillowWriter
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from scipy import interpolate
from matplotlib.animation import FuncAnimation

#%% paths
data_folder = "../../data"
interim_dir = os.path.join(data_folder, "interim")
fig_file = "../../output/figs"
h2o_string = 'retrieval'
log_folder = "../../log"

miawara_Filename_old = 'MIAWARA_concat_H2O_2010_2023_inc_A.nc'
miawara_Filename = 'MIAWARA_concat_H2O_2010_2023_rescaled_inc_A.nc'

miawara_old_fullpath = os.path.join(interim_dir, miawara_Filename_old)
miawara_fullpath = os.path.join(interim_dir, miawara_Filename)