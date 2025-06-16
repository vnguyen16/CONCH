import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import javabridge
import bioformats
import csv
import matplotlib.patches as mpatches

# Constants
# PATCH_SIZE = 224
STRIDE = 224
# TILE_SIZE = 1024
PATCH_SIZE = 224
NUM_PATCHES_PER_TILE = 5
TILE_SIZE = PATCH_SIZE * NUM_PATCHES_PER_TILE  # 1120
SERIES = 10

javabridge.start_vm(class_path=bioformats.JARS)

def compute_statistics2(image, threshold_rng):
    std = np.std(image, axis=-1)
    rng = np.max(image, axis=-1) - np.min(image, axis=-1)
    return np.sum(rng > threshold_rng)

def extract_patches(image, x_offset, y_offset, patch_size=224, stride=224):
    patches = []
    h, w = image.shape[:2]
    for y in range(0, h - patch_size, stride):
        for x in range(0, w - patch_size, stride):
            patch = image[y:y + patch_size, x:x + patch_size, :]
            temp = compute_statistics2(patch, 5)
            if temp.sum() > patch_size * patch_size * 0.3:
                patches.append({"patch": patch, "coordinates": (x + x_offset, y + y_offset)})
    return patches

def load_slide_in_tiles(vsi_path, tile_size=1120, series=8):
    # ImageReader = javabridge.JB_Class("loci.formats.ImageReader")
    # reader = ImageReader()
    ImageReader = javabridge.JClassWrapper("loci.formats.ImageReader")
    reader = ImageReader()

    reader.setId(vsi_path)
    reader.setSeries(series)

    width = reader.getSizeX()
    height = reader.getSizeY()
    channels = reader.getRGBChannelCount()

    tiles = []
    for y in tqdm(range(0, height, tile_size), desc="Reading tiles"):
        for x in range(0, width, tile_size):
            tile_width = min(tile_size, width - x)
            tile_height = min(tile_size, height - y)
            byte_array = reader.openBytes(0, x, y, tile_width, tile_height)
            tile = np.frombuffer(byte_array, dtype=np.uint8).reshape((tile_height, tile_width, channels))
            tiles.append(((x, y), tile))
    reader.close()
    return tiles, (width, height)

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

def show_patch_overlay_on_slide(width, height, patch_info, patch_size=(224, 224), box_color="lime"):
    slide_blank = np.ones((height, width, 3), dtype=np.uint8) * 255
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(slide_blank)
    for patch in patch_info:
        x, y = patch['coordinates']
        rect = mpatches.Rectangle(
            (x, y), patch_size[0], patch_size[1],
            linewidth=1, edgecolor=box_color, facecolor='none'
        )
        ax.add_patch(rect)
    ax.set_title("Overlay of Extracted Patches")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    vsi_path = r"Z:\\mirage\\med-i_data\\Data\\Amoon\\Pathology Raw\\FA scans\\FA 47 B1.vsi"
    output_dir = r"C:\\Users\\Vivian\\Documents\\CONCH\\patches_tiled\\FA_47B1_40x"

    tiles, (w, h) = load_slide_in_tiles(vsi_path, tile_size=TILE_SIZE, series=SERIES)

    all_patches = []
    for (x0, y0), tile in tiles:
        patches = extract_patches(tile, x0, y0, patch_size=PATCH_SIZE, stride=STRIDE)
        all_patches.extend(patches)

    print(f"Extracted {len(all_patches)} relevant patches.")
    # save_patches_as_numpy(all_patches, output_dir)
    # print(f"Patches saved to {output_dir}")

    show_patch_overlay_on_slide(w, h, all_patches, patch_size=(PATCH_SIZE, PATCH_SIZE))
    javabridge.kill_vm()

if __name__ == "__main__":
    main()
