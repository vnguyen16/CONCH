import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def reconstruct_image_with_patch_mask(slide_folder, patch_map_csv, patch_mask_csv, patch_size=(224, 224)):
    """
    Reconstructs an image from patches and overlays:
        - Green boxes for included patches (mask = True)
        - Red boxes for excluded patches (mask = False)
    """

    # Load patch map and patch mask
    patch_map_df = pd.read_csv(patch_map_csv)
    patch_mask_df = pd.read_csv(patch_mask_csv)
    patch_mask = dict(zip(patch_mask_df['patch_file'], patch_mask_df['InsideAnnotation']))

    # Get slide name from folder
    slide_name = os.path.basename(slide_folder)

    # Image dimensions
    max_x = patch_map_df['x'].max() + patch_size[0]
    max_y = patch_map_df['y'].max() + patch_size[1]
    reconstructed_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    overlay_boxes = []

    for _, row in patch_map_df.iterrows():
        patch_file, x, y = row['patch_file'], row['x'], row['y']
        patch_path = os.path.join(slide_folder, patch_file)
        if not os.path.exists(patch_path):
            continue

        patch = np.load(patch_path)
        if patch.ndim == 2:
            patch = np.stack([patch]*3, axis=-1)

        reconstructed_image[y:y + patch_size[1], x:x + patch_size[0], :] = patch

        is_included = patch_mask.get(patch_file, False)
        overlay_color = 'green' if is_included else 'red'
        overlay_boxes.append((x, y, overlay_color))

    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(reconstructed_image)
    ax.set_title(f"Overlay mask for slide: {slide_name}", fontsize=14)
    ax.axis('off')

    for (x, y, color) in overlay_boxes:
        rect = Rectangle((x, y), patch_size[0], patch_size[1], linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.tight_layout()
    plt.show()


def main():
    reconstruct_image_with_patch_mask(
        slide_folder=r"C:\Users\Vivian\Documents\CONCH\all_patches\patches_2.5x\20x\FA\FA 60 B",
        patch_map_csv=r"C:\Users\Vivian\Documents\CONCH\all_patches\patches_2.5x\20x\FA\FA 60 B\patch_map.csv",
        patch_mask_csv=r"C:\Users\Vivian\Documents\CONCH\series9_2.5x_masks\FA 60 B_patch_mask.csv"
    )

if __name__ == "__main__":
    main()