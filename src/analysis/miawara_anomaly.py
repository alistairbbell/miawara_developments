"""
Author: Alistair Bell
Contact: alistair.bell@unibe.ch
Created: 11.09.23
About:
"""

#%%
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


#%%
# Connection details
server_name = 'datatub'  #server descriptor
server_ip = 'datatub.mw.iap.unibe.ch'
username = 'tub_r'
password = ''
client_name = 'user'  # local machine
share_name = 'srv'
server_basepath = 'instruments/miawara/l2/'
data_folder = "../../data"
download_folder = os.path.join(data_folder, "tmp")
temp_file_path = os.path.join(download_folder, "temp_file.nc")
save_output_dir = os.path.join(data_folder, "interim")
output_file_name = 'MIAWARA_concat_H2O_2010_2023.nc'
outFullPath = os.path.join(save_output_dir, output_file_name)
h2o_string = 'retrieval'
log_folder = "../../log"

#%% set up error logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler(os.path.join(log_folder,"miawara_data_retrieve.txt"), maxBytes=1*1024*1024, backupCount=1)
logger.addHandler(handler)

#%%#%% Retrieve the MIAWARA data from the tub server
years = np.arange(2010,2024)
years_s = [str(i) for i in years]

# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True)
conn.connect(server_ip)

# List all files in the root of the share
file_paths_to_merge = []

#variables to drop from the xarray dataset (dataset too big if we keep them)
vars_to_drop = ['y', 'yf', 'tau', 'J'] 

#initialise file iteration number
iterno = 0

for year in years_s:
    svr_data_path = os.path.join(server_basepath, year) 
    files = conn.listPath(share_name, svr_data_path)
    
    for file in files:
        if h2o_string in file.filename and file.filename.endswith(".nc"):
            print(os.path.join(svr_data_path, file.filename)) #for debug
            with open(temp_file_path, 'wb') as local_file:
                conn.retrieveFile(share_name, os.path.join(svr_data_path, file.filename), local_file)
    
            # Load the file into xarray
            ds = xr.open_dataset(temp_file_path, decode_times=False)
            ds_red = ds.drop_vars(vars_to_drop) #drop heavy vars
            
            
            #if not (ds.time.dtype == np.float64 or ds.time.dtype == np.dtype('O')):
            if iterno == 0: #if first file initialise xarray
                concatenated_ds = ds_red
                iterno += 1
            else: #else try to append to existing xarray
                try:
                    concatenated_ds = xr.merge([concatenated_ds, ds_red])
                    iterno += 1
                except Exception as e:
                    print('error encountered')
                    logger.error(traceback.format_exc())
            ds.close() #close temporary datasets
            ds_red.close()
            
concatenated_ds.to_netcdf(outFullPath)
conn.close()
#%%









