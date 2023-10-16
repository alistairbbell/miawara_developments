"""
Author: Alistair Bell
Contact: alistair.bell@unibe.ch
Created: 11.09.23
About: code to get the miawara
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
server_name = 'datatub'  # This can be anything; it's just a descriptor
server_ip = 'datatub.mw.iap.unibe.ch'
username = 'tub_r'
password = ''
client_name = 'user'  # This is the name of the current local machine
share_name = 'srv'
svr_data_path = 'instruments/miawara/l2/'
data_folder = "../../data"
download_folder = os.path.join(data_folder, "tmp")
temp_file_path = os.path.join(download_folder, "temp_file.nc")
save_output_dir = os.path.join(data_folder, "interim")
h2o_string = 'retrieval'
log_folder = "../../log"

#%% set up error logging
logger = logging.getLogger('my_logger')
logger.setLevel(logging.ERROR)
handler = RotatingFileHandler(os.path.join(log_folder,"miawara_data_ret_errlog.txt"), maxBytes=1*1024*1024, backupCount=1)
logger.addHandler(handler)

#%% Retrieve the MLS data from the tub server

# Create connection to server
conn = SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True)
conn.connect(server_ip)

years = np.arange(2015,2024)
years = [str(i) for i in years]

#exclude heavy unneeded variables
exc_variables = ['y', 'yf', 'frequency', 'J', 'tau']

#flag to initialise or append
iterno = 0

for year in years:
    # List all files in the root of the share
    yearpath = os.path.join(svr_data_path, year)
    files = conn.listPath(share_name, yearpath)

    for file in files:
        if h2o_string in file.filename and file.filename.endswith(".nc"):
            print(os.path.join(yearpath, file.filename)) #for debug
            with open(temp_file_path, 'wb') as local_file:
                conn.retrieveFile(share_name, os.path.join(yearpath, file.filename), local_file)
                
            ds = xr.open_dataset(temp_file_path,decode_times=False,
                                 drop_variables=exc_variables)

            if iterno < 1: #if first file initialise xarray
                concatenated_ds = ds
                iterno += 1
            else: #else try to append to existing xarray
                try:
                    concatenated_ds = xr.concat([concatenated_ds, ds], 'time')
                    iterno += 1
                except Exception as e:
                    print('error encountered')
                    logger.error(traceback.format_exc())
            ds.close() #close temporary dataset
            if os.path.exists(temp_file_path):  # Check if file exists before trying to delete
                os.remove(temp_file_path)
    
#organise data by time
concatenated_ds = concatenated_ds.sortby(concatenated_ds['time'])
output_file_name = 'MIAWARA_concat_H2O.nc'
outFullPath = os.path.join(save_output_dir, output_file_name)

encoding = {var: {'zlib': True, 'complevel': 2} for var in concatenated_ds.data_vars}

concatenated_ds.to_netcdf(outFullPath,  encoding=encoding)
print("Merged Miawara file saved to {}".format(outFullPath))

                  
conn.close()
#%% sort the MIAWARA data by time

