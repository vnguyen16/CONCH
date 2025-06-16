import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def visualize_patches(patch_folder, csv_path, num_patches=16):
    """
    Visualize the first few patches.
    """
    df = pd.read_csv(csv_path)
    patches = []

    for idx, row in df.head(num_patches).iterrows():
        patch_file = row['patch_file']
        patch = np.load(os.path.join(patch_folder, patch_file))
        patches.append(patch)

    # Plotting
    cols = 4
    rows = np.ceil(len(patches) / cols).astype(int)
    fig, axes = plt.subplots(rows, cols, figsize=(12, 12))

    for i, ax in enumerate(axes.flatten()):
        if i < len(patches):
            ax.imshow(patches[i])
            ax.set_title(df.iloc[i]['patch_file'], fontsize=8)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.suptitle('Sample Patches', fontsize=16)
    plt.tight_layout()
    plt.show()

def reconstruct_image_from_patches(patch_folder, csv_path, patch_size=(224, 224)):
    """
    Reconstruct the original image from patches using their coordinates.
    """
    df = pd.read_csv(csv_path)

    # Determine the dimensions of the reconstructed image
    max_x = df['x'].max() + patch_size[0]
    max_y = df['y'].max() + patch_size[1]

    reconstructed_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    for _, row in df.iterrows():
        patch_file, x, y = row['patch_file'], row['x'], row['y']
        patch = np.load(os.path.join(patch_folder, patch_file))
        reconstructed_image[y:y + patch_size[1], x:x + patch_size[0], :] = patch

    plt.figure(figsize=(10, 10))
    plt.imshow(reconstructed_image)
    plt.axis('off')
    plt.title('Reconstructed Image', fontsize=16)
    plt.show()

# Example usage:
# patch_folder = "C:\\Users\\Vivian\\Documents\\FA47_B1_level9_numpy"
patch_folder = r"C:\Users\Vivian\Documents\CONCH\patches\20x\FA\FA 88 B"
# patch_folder = r'C:\Users\Vivian\Documents\CONCH\patches_tiled\FA_47B1'
csv_path = os.path.join(patch_folder, "patch_map.csv")

visualize_patches(patch_folder, csv_path, num_patches=16)
reconstruct_image_from_patches(patch_folder, csv_path)
