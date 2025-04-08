# -------------------
# import os
# import subprocess

# # Paths
# bftools_path = r"C:\Users\Vivian\Downloads\bftools\bftools\showinf.bat"  # Path to Bio-Formats tool
# vsi_file = r"C:\Users\Vivian\Documents\slide\FA 57B.vsi"  # VSI file path

# # Command equivalent to: showinf.bat -nopix -omexml "FA 57B.vsi" | findstr "PhysicalSize SizeX SizeY Magnification"
# cmd = f'"{bftools_path}" -nopix -omexml "{vsi_file}" | findstr "PhysicalSize SizeX SizeY Magnification"'

# # Run command with shell=True to support piping (| findstr)
# process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)

# # Capture output
# stdout, stderr = process.communicate()

# # Print filtered metadata
# print(stdout)

# -------------------------------

# THIS WORKS for local files - using bioformats showinf to extract metadata
import os
import subprocess
import pandas as pd
import re
from tqdm import tqdm

# Paths
bftools_path = r"C:\Users\Vivian\Downloads\bftools\bftools\showinf.bat"  # Path to Bio-Formats tool
vsi_folder = r"C:\Users\Vivian\Documents\VSI_Files"  # Folder containing VSI files
# vsi_folder = r"C:\Users\Vivian\Documents\slide" # trying with one slide 
vsi_folder = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans" # trying with remote server
output_csv = "vsi_PT_metadata.csv"  # Output CSV file

# Function to extract metadata from a VSI file
def extract_metadata(vsi_file):
    cmd = f'"{bftools_path}" -nopix -omexml "{vsi_file}" | findstr "PhysicalSize SizeX SizeY Magnification"'
    
    try:
        # Run the command with piping using shell=True
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
        stdout, stderr = process.communicate()
        
        # If there's an error, print and return None
        if stderr:
            print(f"Error processing {vsi_file}: {stderr}")
            return None
        
        # print(stdout)

        # Store metadata
        metadata = {"Filename": os.path.basename(vsi_file)}
        


        # Extract Nominal Magnifications (can be multiple values)
        magnifications = re.findall(r'NominalMagnification="([\d\.]+)"', stdout)
        metadata["Magnifications"] = tuple(map(float, magnifications)) if magnifications else "Unknown"

        # Extract all Pixel Sizes (X, Y) as a list of tuples
        pixel_sizes = re.findall(r'PhysicalSizeX="([\d\.]+)".*?PhysicalSizeY="([\d\.]+)"', stdout, re.DOTALL)
        metadata["Pixel Sizes (X, Y)"] = [tuple(map(float, size)) for size in pixel_sizes] if pixel_sizes else "Unknown"

        # Extract all resolution sizes as a list of tuples (SizeX, SizeY)
        resolution_matches = re.findall(r'SizeX="(\d+)" SizeY="(\d+)"', stdout)
        metadata["Resolution Levels"] = [tuple(map(int, res)) for res in resolution_matches] if resolution_matches else "Unknown"

        # Extract number of pyramid levels
        metadata["Number of Levels"] = len(metadata["Resolution Levels"])

        return metadata
    except subprocess.CalledProcessError:
        print(f"Error reading {vsi_file}")
        return None

# Process all VSI files in the folder
vsi_files = [os.path.join(vsi_folder, f) for f in os.listdir(vsi_folder) if f.endswith(".vsi")]
# metadata_list = [extract_metadata(f) for f in vsi_files if extract_metadata(f)] # og code

# Extract metadata with progress bar 
metadata_list = []
for vsi_file in tqdm(vsi_files, desc="Processing VSI files", unit="file"):
    metadata = extract_metadata(vsi_file)
    if metadata:
        metadata_list.append(metadata)


# Convert to DataFrame and save to CSV
df = pd.DataFrame(metadata_list)
df.to_csv(output_csv, index=False)

print(f"Metadata saved to {output_csv}")


# # problem with parsing the files of the directory, works fine when using a single file. create a loop for the files in the directory
# --------------------------------

# Testing for a single image on remote server - THIS WORKS 

# import os
# import subprocess
# import pandas as pd
# import re

# # Paths
# bftools_path = r"C:\Users\Vivian\Downloads\bftools\bftools\showinf.bat"  # Path to Bio-Formats tool
# vsi_file = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 35 A.vsi"  # Single VSI file path
# output_csv = "PT35_metadata.csv"  # Output CSV file

# # Function to extract metadata from a VSI file
# def extract_metadata(vsi_file):
#     cmd = f'"{bftools_path}" -nopix -omexml "{vsi_file}" | findstr "PhysicalSize SizeX SizeY Magnification"'
    
#     try:
#         # Run the command with piping using shell=True
#         process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, universal_newlines=True)
#         stdout, stderr = process.communicate()
        
#         # If there's an error, print and return None
#         if stderr:
#             print(f"Error processing {vsi_file}: {stderr}")
#             return None
        
#         # Store metadata
#         metadata = {"Filename": os.path.basename(vsi_file)}

#         # Extract Nominal Magnifications (can be multiple values)
#         magnifications = re.findall(r'NominalMagnification="([\d\.]+)"', stdout)
#         metadata["Magnifications"] = tuple(map(float, magnifications)) if magnifications else "Unknown"

#         # Extract all Pixel Sizes (X, Y) as a list of tuples
#         pixel_sizes = re.findall(r'PhysicalSizeX="([\d\.]+)".*?PhysicalSizeY="([\d\.]+)"', stdout, re.DOTALL)
#         metadata["Pixel Sizes (X, Y)"] = [tuple(map(float, size)) for size in pixel_sizes] if pixel_sizes else "Unknown"

#         # Extract all resolution sizes as a list of tuples (SizeX, SizeY)
#         resolution_matches = re.findall(r'SizeX="(\d+)" SizeY="(\d+)"', stdout)
#         metadata["Resolution Levels"] = [tuple(map(int, res)) for res in resolution_matches] if resolution_matches else "Unknown"

#         # Extract number of pyramid levels
#         metadata["Number of Levels"] = len(metadata["Resolution Levels"])

#         return metadata
#     except subprocess.CalledProcessError:
#         print(f"Error reading {vsi_file}")
#         return None

# # Extract metadata from the single VSI file
# metadata = extract_metadata(vsi_file)
# metadata_list = [metadata] if metadata else []

# # Convert to DataFrame and save to CSV
# df = pd.DataFrame(metadata_list)
# df.to_csv(output_csv, index=False)

# print(f"Metadata saved to {output_csv}")
