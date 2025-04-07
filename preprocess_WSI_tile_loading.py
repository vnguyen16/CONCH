# import os
# import csv
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from tqdm import tqdm
# import bioformats
# import javabridge

# # -----------------------------
# # PARAMETERS
# # -----------------------------
# PATCH_SIZE = 224
# STRIDE = 224
# WHITE_THRESHOLD = 0.8
# BORDER_RGB_RANGE = [(0, 0, 0), (70, 60, 65)]
# BORDER_PIXEL_THRESHOLD = 0.01
# TILE_SIZE = 2048  # Size of tiles read from WSI
# SERIES = 7  # Resolution level to read

# # -----------------------------
# # PATCH EXTRACTION LOGIC
# # -----------------------------
# def compute_statistics2(image, threshold_rng):
#     std = np.std(image, axis=-1)
#     rng = np.max(image, axis=-1) - np.min(image, axis=-1)
#     return np.sum(rng > threshold_rng)

# def extract_patches(image, x_offset, y_offset):
#     patches = []
#     h, w = image.shape[:2]

#     for y in range(0, h - PATCH_SIZE, STRIDE):
#         for x in range(0, w - PATCH_SIZE, STRIDE):
#             patch = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE, :]
#             temp = compute_statistics2(patch, 10)
#             if temp > PATCH_SIZE * PATCH_SIZE * 0.3:
#                 patches.append({
#                     "patch": patch,
#                     "coordinates": (x + x_offset, y + y_offset)
#                 })
#     return patches

# # ----------------------------- 
# # PATCH VISUALIZATION
# # -----------------------------
# def visualize_patch_distribution(image, patches):
#     fig, ax = plt.subplots(figsize=(15, 15))
#     ax.imshow(image)
#     for patch_data in patches:
#         x, y = patch_data["coordinates"]
#         rect = mpatches.Rectangle((x, y), PATCH_SIZE, PATCH_SIZE,
#                                   linewidth=0.5, edgecolor="green", facecolor="none")
#         ax.add_patch(rect)
#     plt.axis("off")
#     plt.title("Patch Grid Overlay on WSI")
#     plt.show()

# # -----------------------------
# # PATCH SAVING
# # -----------------------------
# def save_patches(patches, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     for i, patch_data in enumerate(patches):
#         patch = patch_data["patch"]
#         x, y = patch_data["coordinates"]
#         filename = os.path.join(output_dir, f"patch_{i}_x{x}_y{y}.png")
#         cv2.imwrite(filename, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

# def save_patches_as_numpy(patches, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     csv_path = os.path.join(output_dir, "patch_map.csv")
#     with open(csv_path, mode='w', newline='') as csv_file:
#         writer = csv.writer(csv_file)
#         writer.writerow(["patch_file", "x", "y"])
#         for i, patch_data in enumerate(patches):
#             patch = patch_data["patch"]
#             x, y = patch_data["coordinates"]
#             filename = f"patch_{i}_x{x}_y{y}.npy"
#             np.save(os.path.join(output_dir, filename), patch)
#             writer.writerow([filename, x, y])

# # -----------------------------
# # TILE-BASED LOADING FROM BIOFORMATS
# # -----------------------------
# def load_slide_tiled(vsi_path, series, tile_size):
#     from bioformats import ImageReader
#     # Removed unnecessary import of loci.plugins.util
#     reader = ImageReader(vsi_path)
#     reader.setSeries(series)
#     size_x = reader.rdr.getSizeX()
#     size_y = reader.rdr.getSizeY()
#     rgb_image = np.zeros((size_y, size_x, 3), dtype=np.uint8)

#     for y in tqdm(range(0, size_y, tile_size), desc="Reading tiles"):
#         for x in range(0, size_x, tile_size):
#             tile_w = min(tile_size, size_x - x)
#             tile_h = min(tile_size, size_y - y)
#             tile = reader.read(x=x, y=y, width=tile_w, height=tile_h)
#             rgb_image[y:y+tile_h, x:x+tile_w, :] = tile[:, :, :3]

#     return rgb_image

# # -----------------------------
# # MAIN FUNCTION
# # -----------------------------
# def main():
#     # Start JVM
#     javabridge.start_vm(class_path=bioformats.JARS)
#     vsi_file = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 59B.vsi" 
#     output_dir = r"C:\Users\Vivian\Documents\patches\FA59B_level7"

#     try:
#         # Load WSI using tile-based method
#         print("Loading WSI in tiles...")
#         image = load_slide_tiled(vsi_file, series=SERIES, tile_size=TILE_SIZE)
#         print(f"Loaded tiled WSI with shape: {image.shape}")

#         # Extract patches
#         patches = extract_patches(image, 0, 0)
#         print(f"Extracted {len(patches)} relevant patches.")

#         # Visualize patch overlay
#         visualize_patch_distribution(image, patches)

#         # # Save patches (optional)
#         # save_patches(patches, output_dir)
#         # save_patches_as_numpy(patches, output_dir)

#     finally:
#         javabridge.kill_vm()

# if __name__ == "__main__":
#     main()

# -------------------------------------
# import javabridge
# import bioformats
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import cv2

# # from loci.formats import ImageReader
# from bioformats import ImageReader
# # from loci.formats import ChannelSeparator
# # from loci.formats import ImageTools
# # from loci.common import ByteArray
# # from ome.xml.meta import OMEXMLServiceImpl
# # from loci.formats.in import ChannelFiller

# def start_jvm():
#     javabridge.start_vm(class_path=bioformats.JARS)

# def stop_jvm():
#     javabridge.kill_vm()

# def load_tile(reader, series, x, y, width, height):
#     reader.setSeries(series)
#     # openBytesXYWH(seriesIndex, x, y, width, height)
#     bytes_ = reader.openBytes(0, x, y, width, height)
#     return np.frombuffer(bytes_, dtype=np.uint8).reshape(height, width, 3)

# def load_slide_in_tiles(path, series=7, tile_size=1024):
#     reader = ImageReader()
#     reader.setId(path)
#     reader.setSeries(series)
    
#     sizeX = reader.getSizeX()
#     sizeY = reader.getSizeY()
#     print(f"Slide dimensions: {sizeX}x{sizeY}")

#     image = np.zeros((sizeY, sizeX, 3), dtype=np.uint8)

#     for y in range(0, sizeY, tile_size):
#         for x in range(0, sizeX, tile_size):
#             tile_w = min(tile_size, sizeX - x)
#             tile_h = min(tile_size, sizeY - y)

#             tile = reader.openBytes(0, x, y, tile_w, tile_h)
#             tile = np.frombuffer(tile, dtype=np.uint8).reshape(tile_h, tile_w, 3)
#             image[y:y+tile_h, x:x+tile_w, :] = tile

#     reader.close()
#     return image

# # Main execution
# if __name__ == "__main__":
#     vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 59B.vsi"
#     output_path = r"C:\Users\Vivian\Documents\FA59B_thumbnail.png"

#     start_jvm()
    
#     try:
#         image = load_slide_in_tiles(vsi_path, series=7, tile_size=1024)
#         print("Slide loaded. Shape:", image.shape)

#         # Save and visualize thumbnail
#         cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
#         plt.imshow(image)
#         plt.title("Slide Preview (Tiled)")
#         plt.axis("off")
#         plt.show()

#     finally:
#         stop_jvm()

# # -------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import csv
import bioformats
import javabridge

# Parameters
PATCH_SIZE = 224
STRIDE = 224
TILE_SIZE = 1024  # must be manageable to avoid 2GB limits
WHITE_THRESHOLD = 0.8
BORDER_RGB_RANGE = [(0, 0, 0), (70, 60, 65)]
BORDER_PIXEL_THRESHOLD = 0.01

# ------------------------
# Helper Functions
# ------------------------

def compute_statistics2(image, threshold_rng=10):
    std = np.std(image, axis=-1)
    rng = np.max(image, axis=-1) - np.min(image, axis=-1)
    return np.sum(rng > threshold_rng)

def tile_reader_to_numpy(reader, series, tile_size):
    javabridge.attach()

    reader.setSeries(series)
    width = reader.getSizeX()
    height = reader.getSizeY()
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)
            buf = reader.openBytes(0, x, y, w, h)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, -1))
            rgb[y:y+h, x:x+w] = arr[..., :3]  # assume RGB

    javabridge.detach()
    return rgb

def extract_patches(image, patch_size=224, stride=224):
    patches = []
    h, w = image.shape[:2]

    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = image[y:y + patch_size, x:x + patch_size, :]
            temp = compute_statistics2(patch, 10)
            if temp > patch_size * patch_size * 0.3:
                patches.append({"patch": patch, "coordinates": (x, y)})

    return patches

def visualize_patch_distribution(image, patches, patch_size=224):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(image)
    for patch_data in patches:
        x, y = patch_data["coordinates"]
        rect = mpatches.Rectangle((x, y), patch_size, patch_size,
                                  linewidth=0.5, edgecolor="green", facecolor="none")
        ax.add_patch(rect)
    plt.title("Patch Grid Overlay on WSI")
    plt.axis("off")
    plt.show()

def save_patches(patches, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for i, patch_data in enumerate(patches):
        patch = patch_data["patch"]
        x, y = patch_data["coordinates"]
        filename = os.path.join(output_dir, f"patch_{i}_x{x}_y{y}.png")
        cv2.imwrite(filename, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

def save_patches_as_numpy(patches, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "patch_map.csv")
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["patch_file", "x", "y"])
        for i, patch_data in enumerate(patches):
            patch = patch_data["patch"]
            x, y = patch_data["coordinates"]
            patch_filename = f"patch_{i}_x{x}_y{y}.npy"
            np.save(os.path.join(output_dir, patch_filename), patch)
            writer.writerow([patch_filename, x, y])

# ------------------------
# Main Function
# ------------------------

def main():
    import jpype.imports
    from bioformats.omexml import OMEXML
    from loci.formats import ImageReader, ChannelSeparator
    from loci.plugins.util import LociPrefs

    vsi_path = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\FA scans\FA 59B.vsi"
    output_path = r"C:\Users\Vivian\Documents\FA59B_thumbnail.png"


    javabridge.start_vm(class_path=bioformats.JARS)
    reader = ImageReader()
    reader.setId(vsi_path)

    image = tile_reader_to_numpy(reader, series=7, tile_size=TILE_SIZE)
    reader.close()
    javabridge.kill_vm()

    print(f"Loaded tiled WSI image shape: {image.shape}")

    # Process patches
    patches = extract_patches(image)
    print(f"Extracted {len(patches)} relevant patches.")

    # Visualize
    visualize_patch_distribution(image, patches, patch_size=PATCH_SIZE)

    # Save
    # save_patches(patches, output_dir)
    # save_patches_as_numpy(patches, output_dir)

if __name__ == "__main__":
    main()
