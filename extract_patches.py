"""
This script extracts patches from all the slides based on filtered metadata.
"""
# ---------------------------------------------
# # A) loading entire slide into memory then extracting patches (for downsampled images) - this works

# import bioformats
# import javabridge
# import numpy as np
# import os
# import pandas as pd
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from tqdm import tqdm
## from view_image_bioformats import extract_patches, save_patches_as_numpy # used these functions previously
# from load_slide_in_tiles import extract_patches, save_patches_as_numpy

# # -------------------------------
# #  Parameters
# MAGNIFICATION = 40  # max magnification
# METADATA_DIR = "metadata/filtered_slides" # directory to metadata csv files
# OUTPUT_ROOT = f"all_patches/patches_2.5x/{MAGNIFICATION}x" # output directory for patches
# SLIDE_ROOT = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw" # directory to vsi slides

# # -------------------------------
# #  Locate metadata files
# metadata_files = {
#     "FA": os.path.join(METADATA_DIR, f"FA_filtered_slides_{MAGNIFICATION}x.csv"),
#     "PT": os.path.join(METADATA_DIR, f"PT_filtered_slides_{MAGNIFICATION}x.csv")
# }

# # Define class-specific directories
# class_folder = {
#     "FA": "FA scans",
#     "PT": "PT scans"
# }

# # Define series numbers for different magnifications
# series_map = {
#     20: 9,  # Use series 7 for 20x
#     40: 10   # Use series 9 for 40x
# }

# # -------------------------------
# #  Start JVM (only once)
# javabridge.start_vm(class_path=bioformats.JARS)

# # Process both classes (FA & PT)
# for class_name, metadata_file in metadata_files.items():
#     if not os.path.exists(metadata_file):
#         print(f"Skipping {class_name}: No slides found for {MAGNIFICATION}x.")
#         continue

#     print(f"Processing {class_name} slides from {metadata_file}")

#     # Load filtered metadata
#     slides_df = pd.read_csv(metadata_file)

#     for idx, row in tqdm(slides_df.iterrows(), total=len(slides_df)):
#         slide_name = row['Filename']

#         # Ensure the slide belongs to the expected class (FA/PT)
#         if class_name not in slide_name:
#             print(f"Skipping {slide_name}: Does not match class {class_name}.")
#             continue

#         # Extract the correct series number for MAGNIFICATION
#         magnifications = eval(row['Magnifications'])  # Convert string tuple to list
#         if MAGNIFICATION not in magnifications:
#             print(f"Skipping {slide_name}: Magnification {MAGNIFICATION}x not found.")
#             continue

#         #  Assign correct series number based on magnification
#         series_number = series_map[MAGNIFICATION]

#         #  Construct slide path
#         slide_path = os.path.join(SLIDE_ROOT, class_folder[class_name], slide_name)
        
#         # *ADDED*** Define output directory for this slide
#         output_dir = os.path.join(OUTPUT_ROOT, class_name, os.path.splitext(slide_name)[0])

#         # ****ADDED** Check if the slide was already processed (i.e., output directory exists)
#         if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
#             print(f"Skipping {slide_name}: Already processed (patches exist in {output_dir})")
#             continue  # Skip to next slide        

#         # Ensure file exists
#         if not os.path.exists(slide_path):
#             print(f"Slide not found: {slide_path}, skipping.")
#             continue

#         print(f"Processing: {slide_name} | Magnification: {MAGNIFICATION}x | Series: {series_number}")

#         # #  Load the image
#         # image = bioformats.load_image(slide_path, series=series_number, rescale=False)
#         # image_np = np.array(image)

#         # **ADDED** Try to load the image, and skip if it fails
#         try:
#             image_np = bioformats.load_image(slide_path, series=series_number, rescale=False)
#             image_np = np.array(image_np)
#         except Exception as e:
#             print(f"⚠️ Skipping {slide_name} due to error: {e}")
#             continue  # Skip this slide and move to the next one

#         #  Extract patches
#         patches = extract_patches(image_np)

#         #  Define output directory (per slide)
#         output_dir = os.path.join(OUTPUT_ROOT, class_name, os.path.splitext(slide_name)[0])
#         os.makedirs(output_dir, exist_ok=True)

#         #  Save patches
#         save_patches_as_numpy(patches, output_dir)
#         print(f"Processed {slide_name}, saved {len(patches)} patches.")

# # Stop JVM after all slides are processed
# javabridge.kill_vm()
# print("Processing complete!")

# ---------------------------------------------
# B) loading slide in tiles (full res images) and extracting patches
import bioformats
import javabridge
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from load_slide_in_tiles import extract_patches, save_patches_as_numpy, load_slide_in_tiles

# Parameters
MAGNIFICATION = 20 # max magnification 
METADATA_DIR = "metadata/filtered_slides" # directory to metadata csv files
OUTPUT_ROOT = f"patches_tiled/patches_10x/{MAGNIFICATION}x" # output directory for patches
SLIDE_ROOT = r"Z:\\mirage\\med-i_data\\Data\\Amoon\\Pathology Raw" # directory to vsi slides

metadata_files = {
    "FA": os.path.join(METADATA_DIR, f"FA_filtered_slides_{MAGNIFICATION}x.csv"),
    "PT": os.path.join(METADATA_DIR, f"PT_filtered_slides_{MAGNIFICATION}x.csv")
}

class_folder = {
    "FA": "FA scans",
    "PT": "PT scans"
}

# Series numbers for different magnifications
series_map = {
    20: 7,
    40: 8
}

def process_slide(slide_path, output_dir, series_number):
    try:
        tiles, (w, h) = load_slide_in_tiles(slide_path, tile_size=1120, series=series_number)
    except Exception as e:
        print(f"⚠️ Error loading slide {slide_path}: {e}")
        return

    all_patches = []
    for (x0, y0), tile in tiles:
        patches = extract_patches(tile, x0, y0, patch_size=224, stride=224)
        all_patches.extend(patches)

    os.makedirs(output_dir, exist_ok=True)
    save_patches_as_numpy(all_patches, output_dir)
    print(f"✅ Saved {len(all_patches)} patches for {os.path.basename(slide_path)}")

def process_class(class_name, metadata_file):
    if not os.path.exists(metadata_file):
        print(f"Skipping {class_name}: No slides found.")
        return

    slides_df = pd.read_csv(metadata_file)
    print(f"Processing {class_name} slides from {metadata_file}")

    for idx, row in tqdm(slides_df.iterrows(), total=len(slides_df)):
        slide_name = row['Filename']
        if class_name not in slide_name:
            continue

        magnifications = eval(row['Magnifications'])
        if MAGNIFICATION not in magnifications:
            continue

        series_number = series_map[MAGNIFICATION]
        slide_path = os.path.join(SLIDE_ROOT, class_folder[class_name], slide_name)
        output_dir = os.path.join(OUTPUT_ROOT, class_name, os.path.splitext(slide_name)[0])

        if not os.path.exists(slide_path):
            print(f"Slide not found: {slide_path}")
            continue

        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"Skipping {slide_name}: already processed.")
            continue

        print(f"\nProcessing: {slide_name} | Mag: {MAGNIFICATION}x | Series: {series_number}")
        process_slide(slide_path, output_dir, series_number)

def main():
    javabridge.start_vm(class_path=bioformats.JARS)
    for class_name, metadata_file in metadata_files.items():
        process_class(class_name, metadata_file)
    javabridge.kill_vm()
    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()

