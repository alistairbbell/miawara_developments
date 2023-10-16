"""
Author: Alistair Bell
Contact: alistair.bell@unibe.ch
Created: 11.09.23
"""
import subprocess
from smb.SMBConnection import SMBConnection
import os

server_name = 'server_name'
server_ip = 'datatub.mw.iap.unibe.ch'
username = 'tub_r'
password = ''
client_name = 'client_name'
share_name = 'srv'
temp_file_path = "./temp_file.nc"
output_file_path = "./output_file.nc"

conn = SMBConnection(username, password, client_name, server_name, use_ntlm_v2=True)
conn.connect(server_ip)

file_paths = []
files = conn.listPath(share_name, '/atmosphere/AuraMLS/Level2_v5/locations/BERN/')
for file in files:
    if "AuraMLS_L2GP-H2O_v5" in file.filename and file.filename.endswith(".nc"):
        # Download file
        with open(temp_file_path, 'r') as local_file:
            conn.retrieveFile(share_name, '/' + file.filename, local_file)

        # If output_file_path doesn't exist, simply rename temp_file_path to output_file_path
        if not os.path.exists(output_file_path):
            os.rename(temp_file_path, output_file_path)
        else:
            # Concatenate the downloaded file to the output file
            cmd = ["ncrcat", output_file_path, temp_file_path, "-o", "temp_output.nc"]
            subprocess.run(cmd, check=True)

            # Replace output_file with the concatenated file
            os.rename("temp_output.nc", output_file_path)

            # Remove the downloaded temp file
            os.remove(temp_file_path)

conn.close()
