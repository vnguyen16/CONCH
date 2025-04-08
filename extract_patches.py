"""
This script extracts patches from all the slides based on filtered metadata.
"""

# import bioformats
# import javabridge
# import numpy as np
# import os
# import pandas as pd
# from view_image_bioformats import extract_patches, save_patches_as_numpy

# # Example parameters
# magnification = 20
# class_name = "FA"
# filtered_metadata_path = f'metadata/filtered_slides/{class_name}_filtered_slides_{magnification}x.csv'
# output_root = f'patches/{magnification}x'

# slides_df = pd.read_csv(filtered_metadata_path)
# slide_root = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw"

# # Assuming class_name is either "FA" or "PT"
# class_folder = {
#     "FA": "FA scans",
#     "PT": "PT scans"
# }



# # Start JVM once, outside the loop (performance optimization)
# javabridge.start_vm(class_path=bioformats.JARS)

# for idx, row in slides_df.iterrows():
#     slide_name = row['Filename']
#     series_number = row['Magnifications'].index(magnification)

#     # slide_path = os.path.join("your_slide_root_folder", slide_name)

#     # When loading each slide
#     slide_path = os.path.join(slide_root, class_folder[class_name], slide_name)

#     image = bioformats.load_image(slide_path, series=series_number, rescale=False)
#     image_np = np.array(image)

#     patches = extract_patches(image_np)

#     output_dir = os.path.join(output_root, os.path.splitext(slide_name)[0])
#     save_patches_as_numpy(patches, output_dir)
#     print(f"Processed {slide_name}.")

# javabridge.kill_vm()

# ---------------------------------------------
# import bioformats
# import javabridge
# import numpy as np
# import os
# import pandas as pd
# from view_image_bioformats import extract_patches, save_patches_as_numpy

# # Example parameters
# magnification = 20
# metadata_dir = "metadata/filtered_slides"

# # Determine metadata file paths for both classes
# metadata_files = {
#     "FA": os.path.join(metadata_dir, f"FA_filtered_slides_{magnification}x.csv"),
#     "PT": os.path.join(metadata_dir, f"PT_filtered_slides_{magnification}x.csv")
# }

# # Ensure at least one metadata file exists
# existing_files = [f for f in metadata_files.values() if os.path.exists(f)]
# if not existing_files:
#     raise FileNotFoundError(f"No metadata files found for magnification {magnification}x in {metadata_dir}")

# # Process both classes if their metadata exists
# for class_name, metadata_file in metadata_files.items():
#     if not os.path.exists(metadata_file):
#         print(f"Skipping {class_name}: No slides found for {magnification}x.")
#         continue

#     print(f"Processing class: {class_name} from {metadata_file}")

#     slides_df = pd.read_csv(metadata_file)
#     slide_root = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw"

#     # Define class-specific directories
#     class_folder = {
#         "FA": "FA scans",
#         "PT": "PT scans"
#     }

#     # Start JVM once, outside the loop (performance optimization)
#     javabridge.start_vm(class_path=bioformats.JARS)

#     for idx, row in slides_df.iterrows():
#         slide_name = row['Filename']
        
#         # Ensure the slide contains the expected class name
#         if class_name not in slide_name:
#             print(f"Skipping {slide_name}: Does not match expected class {class_name}.")
#             continue
        
#         # Find correct series number based on magnification
#         magnifications = eval(row['Magnifications'])  # Convert string to tuple
#         if magnification not in magnifications:
#             print(f"Skipping {slide_name}: Magnification {magnification}x not found.")
#             continue

#         series_number = magnifications.index(magnification)
        
#         # Construct full slide path
#         slide_path = os.path.join(slide_root, class_folder[class_name], slide_name)

#         # Ensure the slide file exists before processing
#         if not os.path.exists(slide_path):
#             print(f"Warning: Slide file not found: {slide_path}, skipping.")
#             continue

#         print(f"Processing: {slide_name} | Series: {series_number}")

#         # Load the image
#         image = bioformats.load_image(slide_path, series=series_number, rescale=False)
#         image_np = np.array(image)

#         # Extract patches
#         patches = extract_patches(image_np)

#         # Define output directory
#         output_root = f'patches/{magnification}x/{class_name}'
#         output_dir = os.path.join(output_root, os.path.splitext(slide_name)[0])
#         os.makedirs(output_dir, exist_ok=True)

#         # Save patches
#         save_patches_as_numpy(patches, output_dir)
#         print(f"Processed {slide_name}, saved {len(patches)} patches.")

#     # Stop JVM after processing all slides in the class
#     javabridge.kill_vm()

# print("Processing complete.")

# ---------------------------------------------

import bioformats
import javabridge
import numpy as np
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
from view_image_bioformats import extract_patches, save_patches_as_numpy

# -------------------------------
#  Parameters
MAGNIFICATION = 20  # Change this to 40 if needed
METADATA_DIR = "metadata/filtered_slides"
OUTPUT_ROOT = f"patches_series8/{MAGNIFICATION}x"
SLIDE_ROOT = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw"

# -------------------------------
#  Locate metadata files
metadata_files = {
    "FA": os.path.join(METADATA_DIR, f"FA_filtered_slides_{MAGNIFICATION}x.csv"),
    "PT": os.path.join(METADATA_DIR, f"PT_filtered_slides_{MAGNIFICATION}x.csv")
}

# Define class-specific directories
class_folder = {
    "FA": "FA scans",
    "PT": "PT scans"
}

# Define series numbers for different magnifications
series_map = {
    20: 7,  # Use series 7 for 20x
    40: 9   # Use series 9 for 40x
}

# -------------------------------
#  Start JVM (only once)
javabridge.start_vm(class_path=bioformats.JARS)

# Process both classes (FA & PT)
for class_name, metadata_file in metadata_files.items():
    if not os.path.exists(metadata_file):
        print(f"Skipping {class_name}: No slides found for {MAGNIFICATION}x.")
        continue

    print(f"Processing {class_name} slides from {metadata_file}")

    # Load filtered metadata
    slides_df = pd.read_csv(metadata_file)

    for idx, row in tqdm(slides_df.iterrows(), total=len(slides_df)):
        slide_name = row['Filename']

        # Ensure the slide belongs to the expected class (FA/PT)
        if class_name not in slide_name:
            print(f"Skipping {slide_name}: Does not match class {class_name}.")
            continue

        # Extract the correct series number for MAGNIFICATION
        magnifications = eval(row['Magnifications'])  # Convert string tuple to list
        if MAGNIFICATION not in magnifications:
            print(f"Skipping {slide_name}: Magnification {MAGNIFICATION}x not found.")
            continue

        #  Assign correct series number based on magnification
        series_number = series_map[MAGNIFICATION]

        #  Construct slide path
        slide_path = os.path.join(SLIDE_ROOT, class_folder[class_name], slide_name)
        
        # *ADDED*** Define output directory for this slide
        output_dir = os.path.join(OUTPUT_ROOT, class_name, os.path.splitext(slide_name)[0])

        # ****ADDED** Check if the slide was already processed (i.e., output directory exists)
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            print(f"Skipping {slide_name}: Already processed (patches exist in {output_dir})")
            continue  # Skip to next slide        

        # Ensure file exists
        if not os.path.exists(slide_path):
            print(f"Slide not found: {slide_path}, skipping.")
            continue

        print(f"Processing: {slide_name} | Magnification: {MAGNIFICATION}x | Series: {series_number}")

        # #  Load the image
        # image = bioformats.load_image(slide_path, series=series_number, rescale=False)
        # image_np = np.array(image)

        # **ADDED** Try to load the image, and skip if it fails
        try:
            image_np = bioformats.load_image(slide_path, series=series_number, rescale=False)
            image_np = np.array(image_np)
        except Exception as e:
            print(f"⚠️ Skipping {slide_name} due to error: {e}")
            continue  # Skip this slide and move to the next one

        #  Extract patches
        patches = extract_patches(image_np)

        #  Define output directory (per slide)
        output_dir = os.path.join(OUTPUT_ROOT, class_name, os.path.splitext(slide_name)[0])
        os.makedirs(output_dir, exist_ok=True)

        #  Save patches
        save_patches_as_numpy(patches, output_dir)
        print(f"Processed {slide_name}, saved {len(patches)} patches.")

# Stop JVM after all slides are processed
javabridge.kill_vm()
print("Processing complete!")
