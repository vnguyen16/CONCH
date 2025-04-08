"""
This script contains programs to:
    1. Visualize patches from a folder + reconstruct the original image from patches.
    2. Reconstruct an image from patches and overlay predictions.
    3. Compute slide-level accuracy from patch-level predictions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Rectangle

# ---------------------------------------------
# 1. Visualize patches and reconstruct image
# ---------------------------------------------

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

# ---------------------------------------------
# 2. Reconstruct image with predictions overlaid
# ---------------------------------------------

def reconstruct_image_with_predictions(slide_folder, patch_map_csv, predictions_df, patch_size=(224, 224)):
    """
    Reconstructs an image from patches and applies red overlays for FA (class 0)
    and blue overlays for PT (class 1).
    """

    # Load patch map for the specific slide
    patch_map_df = pd.read_csv(patch_map_csv)

    # Extract slide name (e.g., "FA 56B" or "PT 41 B") from folder path
    slide_name = os.path.basename(slide_folder)

    # Filter predictions only for this slide
    slide_predictions = predictions_df[predictions_df['Patch Path'].str.contains(slide_name, regex=False)]

    if slide_predictions.empty:
        print(f"Skipping {slide_name}: No predictions found.")
        return

    # Convert predictions into a dictionary for fast lookup
    predictions_dict = {os.path.basename(row['Patch Path']): (row['Predicted'], row['True Label'])
                        for _, row in slide_predictions.iterrows()}

    # Determine reconstructed image dimensions
    max_x = patch_map_df['x'].max() + patch_size[0]
    max_y = patch_map_df['y'].max() + patch_size[1]

    reconstructed_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)

    # Initialize overlay positions
    overlay_boxes = []

    for _, row in patch_map_df.iterrows():
        patch_file, x, y = row['patch_file'], row['x'], row['y']

        # Load the patch
        patch_path = os.path.join(slide_folder, patch_file)
        if not os.path.exists(patch_path):
            continue  # Skip missing files
        patch = np.load(patch_path)

        # Ensure patches are RGB
        if patch.ndim == 2:  # Convert grayscale to RGB
            patch = np.stack([patch]*3, axis=-1)

        # Add patch to the reconstructed image
        reconstructed_image[y:y + patch_size[1], x:x + patch_size[0], :] = patch

        # Check if the patch has a prediction
        if patch_file in predictions_dict:
            predicted, true_label = predictions_dict[patch_file]

            # Assign color based on FA (0) / PT (1)
            if predicted == 0:
                overlay_color = 'red'    # FA
            elif predicted == 1:
                overlay_color = 'blue'   # PT
            else:
                overlay_color = 'gray'   # Unexpected class fallback

            overlay_boxes.append((x, y, overlay_color))

    # Display the reconstructed image
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(reconstructed_image)
    ax.axis('off')
    ax.set_title(f"Reconstructed Slide: {slide_name}", fontsize=16)

    # Draw overlay boxes
    for (x, y, color) in overlay_boxes:
        rect = Rectangle((x, y), patch_size[0], patch_size[1], linewidth=1, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    plt.show()


def process_selected_slides(root_folder, predictions_csv, patch_size=(224, 224)):
    """
    Processes only the slides found in the predictions CSV.
    """

    # Load full predictions CSV
    predictions_df = pd.read_csv(predictions_csv)

    # Extract unique slide names from predictions
    unique_slides = set(predictions_df['Patch Path'].apply(lambda x: os.path.basename(os.path.dirname(x))))

    # Iterate over magnification levels (20x, 40x)
    for magnification_level in ["20x", "40x"]:
        magnification_path = os.path.join(root_folder, magnification_level)
        if not os.path.exists(magnification_path):
            continue

        # Iterate over FA and PT directories
        for category in ["FA", "PT"]:
            category_path = os.path.join(magnification_path, category)
            if not os.path.exists(category_path):
                continue

            # Iterate over slide folders and process only those in the predictions CSV
            for slide_name in os.listdir(category_path):
                slide_folder = os.path.join(category_path, slide_name)
                if not os.path.isdir(slide_folder):  # Skip non-folder files
                    continue

                if slide_name not in unique_slides:
                    continue  # Skip slides not in the predictions CSV

                patch_map_csv = os.path.join(slide_folder, "patch_map.csv")
                if os.path.exists(patch_map_csv):
                    print(f"Processing slide: {slide_name} ({magnification_level}, {category})")
                    reconstruct_image_with_predictions(slide_folder, patch_map_csv, predictions_df, patch_size)

# ----------------------------------------------
# 3. Compute slide accuracy from predictions
# ----------------------------------------------


def compute_slide_accuracy(csv_path, output_csv_path=None, save_csv=True):
    """
    Computes accuracy per slide from a CSV file with patch-level predictions.

    Parameters:
        csv_path (str): Path to the CSV file containing 'Patch Path', 'Predicted', and 'True Label' columns.
        output_csv_path (str, optional): Path to save the slide-level accuracy CSV. Required if save_csv=True.
        save_csv (bool): Whether to save the output as a CSV file.

    Returns:
        pd.DataFrame: DataFrame containing 'Slide', 'Accuracy', and 'Accuracy (%)'.
    """
    df = pd.read_csv(csv_path)

    # Extract slide name from patch path (second-to-last directory)
    df['Slide'] = df['Patch Path'].apply(lambda x: os.path.normpath(x).split(os.sep)[-2])

    # Compare predicted vs true label
    df['Correct'] = df['Predicted'] == df['True Label']

    # Group by slide and calculate mean accuracy
    accuracy_per_slide = df.groupby('Slide')['Correct'].mean().reset_index()
    accuracy_per_slide.columns = ['Slide', 'Accuracy']
    accuracy_per_slide['Accuracy (%)'] = accuracy_per_slide['Accuracy'] * 100

    # Save to CSV if enabled
    if save_csv:
        if output_csv_path is None:
            raise ValueError("output_csv_path must be specified if save_csv is True.")
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        accuracy_per_slide.to_csv(output_csv_path, index=False)
        print(f"Slide-level accuracy saved to {output_csv_path}")

    return accuracy_per_slide


def main():
    # --------------- 1. visualize/reconstruct patches ---------------------
    # patch_folder = "C:\\Users\\Vivian\\Documents\\FA47_B1_level9_numpy"
    # patch_folder = r"C:\Users\Vivian\Documents\CONCH\patches\20x\FA\FA 88 B"
    patch_folder = r'C:\Users\Vivian\Documents\CONCH\patches_annotated\20x\FA\FA 62 B'
    csv_path = os.path.join(patch_folder, "patch_map.csv")

    visualize_patches(patch_folder, csv_path, num_patches=16)
    reconstruct_image_from_patches(patch_folder, csv_path)

    # ------------- 2. reconstruct image with predictions overlaid ---------------------
    # root_folder = r"C:\Users\Vivian\Documents\CONCH\patches"
    # # predictions_csv = r'C:\Users\Vivian\Documents\CONCH\patch_predictions\patient_split_UNI70_linprob.csv'
    # # predictions_csv = r'C:\Users\Vivian\Documents\CONCH\patch_predictions\annotated\UNI_linprob_ann_test.csv'
    # # predictions_csv = r"C:\Users\Vivian\Documents\CONCH\patch_predictions\annotated\CONCH_linprob_ann_test.csv"
    # predictions_csv = r'C:\Users\Vivian\Documents\CONCH\patch_predictions\UNI90_test_ann_only.csv'

    # process_selected_slides(root_folder, predictions_csv)

    # ---------------- 3. compute slide accuracy ---------------------
    # csv_path = r"C:\Users\Vivian\Documents\CONCH\patch_predictions\patient_split_CONCH70.csv"
    # output_path = r"C:\Users\Vivian\Documents\CONCH\slide_acc\conch70_patient.csv"

    # acc_df = compute_slide_accuracy(csv_path, output_path, save_csv=True)
    # # acc_df = compute_slide_accuracy(csv_path, save_csv=False) # Just to display without saving

    # print(acc_df.sort_values(by='Accuracy (%)', ascending=False))


if __name__ == "__main__":
    main()
