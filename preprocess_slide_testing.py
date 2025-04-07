import os
import bioformats
import javabridge
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm.notebook import tqdm
import matplotlib.patches as mpatches  

# Parameters
PATCH_SIZE = 224  # Patch size (width, height)
STRIDE = 224  # Sliding window stride
WHITE_THRESHOLD = 0.8  # Max ratio of white pixels to exclude patch
BORDER_RGB_RANGE = [(0, 0, 0), (70, 60, 65)]  # RGB range for border-like regions
BORDER_PIXEL_THRESHOLD = 0.01  # Max proportion of border-like pixels to exclude patch
TILE_SIZE = 1024  # Load the image in smaller tiles to avoid memory issues

def start_jvm():
    """ Start Java Virtual Machine for Bio-Formats """
    javabridge.start_vm(class_path=bioformats.JARS)

def stop_jvm():
    """ Stop Java Virtual Machine """
    javabridge.kill_vm()

def load_slide_tiled(vsi_file, series=7, tile_size=1024):
    """
    Load a VSI slide in smaller tiles to avoid memory errors.
    Args:
        vsi_file (str): Path to the Whole Slide Image.
        series (int): Resolution level (default=7).
        tile_size (int): Size of tile to load at a time (default=1024).
    Returns:
        numpy.array: Loaded WSI as a NumPy array.
    """
    start_jvm()  # Start JVM for Bio-Formats
    image_reader = bioformats.ImageReader(vsi_file)

    # Get image dimensions at selected series level
    width = image_reader.rdr.getSizeX()
    height = image_reader.rdr.getSizeY()
    num_channels = image_reader.rdr.getSizeC()

    # Create empty NumPy array to store the slide
    image = np.zeros((height, width, num_channels), dtype=np.uint8)

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            # Compute tile dimensions
            tile_width = min(tile_size, width - x)
            tile_height = min(tile_size, height - y)

            # Read the tile using openBytes()
            raw_tile = image_reader.read(XYWH=(x, y, tile_width, tile_height), series=series)
            tile = np.array(raw_tile, dtype=np.uint8)

            # Place tile in the correct position in the full image
            image[y:y+tile_height, x:x+tile_width] = tile

    stop_jvm()  # Stop JVM after reading
    return image


# def compute_statistics(image):
#     """
#     Compute the ratio of white pixels and border-like pixels in the patch.
#     Args:
#         image (numpy.array): Patch of the image (WxHxC).
#     Returns:
#         tuple: (white_pixel_ratio, border_pixel_ratio)
#     """
#     num_pixels = image.shape[0] * image.shape[1]
    
#     # Calculate white pixel ratio
#     summed_matrix = np.sum(image, axis=-1)  # Sum of RGB channels
#     white_pixels = summed_matrix > 620  # Threshold for white pixels (RGB ~ [255, 255, 255])
#     num_white_pixels = np.count_nonzero(white_pixels)
#     ratio_white_pixels = num_white_pixels / num_pixels

#     # Calculate border-like pixel ratio
#     lower_range, upper_range = BORDER_RGB_RANGE
#     border_pixels = np.all(
#         (image >= lower_range) & (image <= upper_range),
#         axis=-1
#     )
#     num_border_pixels = np.count_nonzero(border_pixels)
#     ratio_border_pixels = num_border_pixels / num_pixels

#     return ratio_white_pixels, ratio_border_pixels

# def extract_patches(image, patch_size=224, stride=224, white_threshold=0.8, border_pixel_threshold=0.2):
#     """
#     Extract and filter patches based on white pixel ratio and border pixel ratio.
#     Args:
#         image (numpy.array): Input WSI image (RGB).
#         patch_size (int): Size of each patch.
#         stride (int): Stride for sliding window.
#         white_threshold (float): Maximum allowed ratio of white pixels in a patch.
#         border_pixel_threshold (float): Maximum allowed ratio of border-like pixels in a patch.
#     Returns:
#         list: Relevant patches with their coordinates.
#     """
#     patches = []
#     h, w = image.shape[:2]

#     for y in range(0, h - patch_size, stride):
#         for x in range(0, w - patch_size, stride):
#             patch = image[y:y + patch_size, x:x + patch_size, :]

#             # Compute statistics
#             ratio_white, ratio_border = compute_statistics(patch)

#             # Keep patch if it has **low white pixels and low border pixels**
#             if ratio_white < white_threshold and ratio_border < border_pixel_threshold:
#                 patches.append({"patch": patch, "coordinates": (x, y)})

#     return patches


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

def main():
    """ Main Function for WSI Processing """
    # Set VSI File Path
    vsi_file = "Z:/mirage/med-i_data/Data/Amoon/Pathology Raw/FA scans/FA 60 B.vsi"
    output_dir = "C:/Users/Vivian/Documents/FA60_B1_level7_numpy"

    print("\n--- Loading Slide ---")
    image = load_slide_tiled(vsi_file, series=7, tile_size=TILE_SIZE)
    
     # Display a thumbnail to check the slide
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Thumbnail of Loaded WSI")
    plt.axis("off")
    plt.show()

    print(f"Loaded WSI with shape: {image.shape}")

    print("\n--- Extracting Patches ---")
    # Extract relevant patches
    patches = extract_patches(
        image,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        white_threshold=WHITE_THRESHOLD,
        border_pixel_threshold=BORDER_PIXEL_THRESHOLD
    )
    print(f"Extracted {len(patches)} relevant patches.")

    print("\n--- Visualizing Patch Distribution ---")
    visualize_patch_distribution(image, patches, patch_size=PATCH_SIZE)

    # print("\n--- Saving Patches ---")
    # save_patches(patches, output_dir)
    # print(f"Patches saved to {output_dir}")

if __name__ == "__main__":
    main()
