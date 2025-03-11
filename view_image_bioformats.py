# # Loading image with bioformats and coverting to numpy arrray - THIS WORKS

# import bioformats
# import javabridge
# import numpy as np
# import matplotlib.pyplot as plt

# # Start Java Virtual Machine
# javabridge.start_vm(class_path=bioformats.JARS)

# # Suppress Bio-Formats debug logs
# # javabridge.call("loci.common.DebugTools", "setRootLevel", "Ljava/lang/String;", "ERROR")

# # Read the VSI image
# # vsi_file = "C:/Users/Vivian/Documents/slide/PT 96 A1.vsi"
# # vsi_file = "Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 54 B1.vsi" # testing with slide from serevr
# # vsi_file = "Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 47 B1.vsi" # 40x mag - use series 9?
# vsi_file = "Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi" # 20x mag - use series 7?

# # image = bioformats.load_image(vsi_file)
# # metadata = bioformats.get_omexml_metadata(vsi_file)

# image = bioformats.load_image(vsi_file, series=7, rescale=False) # Highest resolution that could be loaded = 7 (FA 57B (20701, 28980, 3) - 2nd highest size)

# # Convert to NumPy array
# image_np = np.array(image)

# # Stop JVM
# javabridge.kill_vm()


# ---------------------------------------------

# using the numpy array created with the bioformats library - THIS WORKS

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import matplotlib.patches as mpatches  

# Parameters
PATCH_SIZE = 224  # Patch size (width, height)
STRIDE = 224  # Sliding window stride
WHITE_THRESHOLD = 0.8  # Max ratio of white pixels to exclude patch
BORDER_RGB_RANGE = [(0, 0, 0), (70, 60, 65)]  # RGB range for border-like regions
BORDER_PIXEL_THRESHOLD = 0.01  # Max proportion of border-like pixels to exclude patch

def compute_statistics2(image, threshold_rng):
    std = np.std(image, axis=-1)
    rng = np.max(image, axis=-1) - np.min(image, axis=-1)
    return np.sum(rng > threshold_rng)


def extract_patches(image, patch_size=224, stride=224, white_threshold=0.8, border_pixel_threshold=0.2):
    """
    Extract and filter patches based on white pixel ratio and border pixel ratio.
    Args:
        image (numpy.array): Input WSI image (RGB).
        patch_size (int): Size of each patch.
        stride (int): Stride for sliding window.
        white_threshold (float): Maximum allowed ratio of white pixels in a patch.
        border_pixel_threshold (float): Maximum allowed ratio of border-like pixels in a patch.
    Returns:
        list: Relevant patches with their coordinates.
    """
    patches = []
    h, w = image.shape[:2]

    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = image[y:y + patch_size, x:x + patch_size, :]

            # Second approach 
            temp = compute_statistics2(patch, 10) # determine if pixel is colour or not # og 40
            if temp.sum() > patch_size * patch_size * 0.3: # og 0.4 # count number of colour pixels, see percentage of patch that is colour or white
                patches.append({"patch": patch, "coordinates": (x, y)})

    return patches

def visualize_patch_distribution(image, patches, patch_size=224):
    """
    Overlay extracted patch locations as a grid on the original image.
    Args:
        image (numpy.array): Input WSI image (RGB).
        patches (list): List of selected patches with their coordinates.
        patch_size (int): Size of each patch.
    """
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Show the original image
    ax.imshow(image)

    # Overlay grid (patch outlines)
    for patch_data in patches:
        x, y = patch_data["coordinates"]
        rect = mpatches.Rectangle(  
            (x, y),  # Top-left corner
            patch_size,  # Width
            patch_size,  # Height
            linewidth=0.5,  # Thin border
            edgecolor="green",  # Grid color
            facecolor="none"  # Transparent fill
        )
        ax.add_patch(rect)

    plt.title("Patch Grid Overlay on WSI")
    plt.axis("off")  # Hide axes for a cleaner visualization
    plt.show()

def save_patches(patches, output_dir):
    """
    Save selected patches to disk.
    Args:
        patches (list): List of patches with their coordinates.
        output_dir (str): Directory to save patches.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, patch_data in enumerate(patches):
        patch = patch_data["patch"]
        x, y = patch_data["coordinates"]
        filename = os.path.join(output_dir, f"patch_{i}_x{x}_y{y}.png")
        cv2.imwrite(filename, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

# ----------------------------------------------
# attempting to save patches as numpy arrays
import csv
def save_patches_as_numpy(patches, output_dir):
    """
    Save selected patches as NumPy arrays and create a CSV map inside the output directory.
    Args:
        patches (list): List of patches with their coordinates.
        output_dir (str): Directory to save patches and CSV map.
    """
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "patch_map.csv")

    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["patch_file", "x", "y"])  # Header

        for i, patch_data in enumerate(patches):
            patch = patch_data["patch"]
            x, y = patch_data["coordinates"]
            patch_filename = f"patch_{i}_x{x}_y{y}.npy"
            patch_path = os.path.join(output_dir, patch_filename)

            # Save the patch as a .npy file
            np.save(patch_path, patch)

            # Write mapping to CSV
            writer.writerow([patch_filename, x, y])


# ----------------------------------------------
# # Main Workflow
# output_dir = "C:\\Users\\Vivian\\Documents\\FA57_B1_level7_numpy"

# # Use the numpy array created with the bioformats library
# image = image_np
# print(f"Loaded WSI with shape: {image.shape}")

# # Extract relevant patches
# patches = extract_patches(
#     image,
#     patch_size=PATCH_SIZE,
#     stride=STRIDE,
#     white_threshold=WHITE_THRESHOLD,
#     border_pixel_threshold=BORDER_PIXEL_THRESHOLD
# )
# print(f"Extracted {len(patches)} relevant patches.")

# # Visualize patch distribution
# visualize_patch_distribution(image, patches, patch_size=PATCH_SIZE)

# Save patches
# save_patches(patches, output_dir)
# print(f"Patches saved to {output_dir}")

# ----------------------------------------------
# Saving patches as NumPy arrays
# save_patches_as_numpy(patches, output_dir)
# print(f"Patches and mapping saved to {output_dir}")

# ----------------------------------------------

def main():
        
    # Loading image with bioformats and coverting to numpy arrray - THIS WORKS
    import bioformats
    import javabridge
    import numpy as np
    import matplotlib.pyplot as plt

    # Start Java Virtual Machine
    javabridge.start_vm(class_path=bioformats.JARS)

    # Suppress Bio-Formats debug logs
    # javabridge.call("loci.common.DebugTools", "setRootLevel", "Ljava/lang/String;", "ERROR")

    # Read the VSI image
    # vsi_file = "C:/Users/Vivian/Documents/slide/PT 96 A1.vsi"
    # vsi_file = "Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 54 B1.vsi" # testing with slide from serevr
    # vsi_file = "Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 47 B1.vsi" # 40x mag - use series 9?
    # vsi_file = "Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 57B.vsi" # 20x mag - use series 7?
    # vsi_file = "Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 60 B.vsi"
    vsi_file = "z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans\PT 74 B2.vsi" # 40x

    # image = bioformats.load_image(vsi_file)
    # metadata = bioformats.get_omexml_metadata(vsi_file)

    image = bioformats.load_image(vsi_file, series=9, rescale=False) # Highest resolution that could be loaded = 7 (FA 57B (20701, 28980, 3) - 2nd highest size)

    # Convert to NumPy array
    image_np = np.array(image)

    # Stop JVM
    javabridge.kill_vm()

    # --------------------------------------------- 

    # Main Workflow
    output_dir = r"C:\Users\Vivian\Documents\CONCH\patches_png\PT74_B2_level9"

    # Use the numpy array created with the bioformats library
    image = image_np
    print(f"Loaded WSI with shape: {image.shape}")

    # Extract relevant patches
    patches = extract_patches( 
        image,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        white_threshold=WHITE_THRESHOLD,
        border_pixel_threshold=BORDER_PIXEL_THRESHOLD
    )
    print(f"Extracted {len(patches)} relevant patches.")

    # Visualize patch distribution
    visualize_patch_distribution(image, patches, patch_size=PATCH_SIZE)

    # # Save patches
    # save_patches(patches, output_dir)
    # print(f"Patches saved to {output_dir}")

    # # Saving patches as NumPy arrays
    # save_patches_as_numpy(patches, output_dir)
    # print(f"Patches and mapping saved to {output_dir}")

if __name__ == "__main__":
    main()