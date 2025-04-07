"""
Reconstructing slides with patches
"""
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib.patches import Rectangle

# def reconstruct_image_with_predictions(slide_folder, patch_map_csv, predictions_df, patch_size=(224, 224)):
#     """
#     Reconstructs an image from patches and applies green overlays for correct predictions
#     and red overlays for incorrect predictions.
#     """

#     # Load patch map for the specific slide
#     patch_map_df = pd.read_csv(patch_map_csv)

#     # Extract slide name (e.g., "FA 56B" or "PT 41 B") from folder path
#     slide_name = os.path.basename(slide_folder)

#     # Filter predictions only for this slide
#     slide_predictions = predictions_df[predictions_df['Patch Path'].str.contains(slide_name, regex=False)]

#     # Convert predictions into a dictionary for fast lookup
#     predictions_dict = {os.path.basename(row['Patch Path']): (row['Predicted'], row['True Label'])
#                         for _, row in slide_predictions.iterrows()}

#     # Determine reconstructed image dimensions
#     max_x = patch_map_df['x'].max() + patch_size[0]
#     max_y = patch_map_df['y'].max() + patch_size[1]

#     reconstructed_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)

#     # Initialize overlay positions
#     overlay_boxes = []

#     for _, row in patch_map_df.iterrows():
#         patch_file, x, y = row['patch_file'], row['x'], row['y']

#         # Load the patch
#         patch_path = os.path.join(slide_folder, patch_file)
#         if not os.path.exists(patch_path):
#             continue  # Skip missing files
#         patch = np.load(patch_path)

#         # Ensure patches are RGB
#         if patch.ndim == 2:  # Convert grayscale to RGB
#             patch = np.stack([patch]*3, axis=-1)

#         # Add patch to the reconstructed image
#         reconstructed_image[y:y + patch_size[1], x:x + patch_size[0], :] = patch

#         # Check if the patch has a prediction
#         if patch_file in predictions_dict:
#             predicted, true_label = predictions_dict[patch_file]
#             overlay_color = 'green' if predicted == true_label else 'red'
#             overlay_boxes.append((x, y, overlay_color))

#     # Display the reconstructed image
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(reconstructed_image)
#     ax.axis('off')
#     ax.set_title(f"Reconstructed Slide: {slide_name}", fontsize=16)

#     # Draw overlay boxes
#     for (x, y, color) in overlay_boxes:
#         rect = Rectangle((x, y), patch_size[0], patch_size[1], linewidth=2, edgecolor=color, facecolor='none')
#         ax.add_patch(rect)

#     plt.show()


# def process_all_slides(root_folder, predictions_csv, patch_size=(224, 224)):
#     """
#     Iterates through all magnification levels (20x, 40x), FA and PT directories,
#     and reconstructs each slide separately using its corresponding patch map and predictions.
#     """

#     # Load full predictions CSV
#     predictions_df = pd.read_csv(predictions_csv)

#     # Iterate over magnification levels (20x, 40x)
#     for magnification_level in ["20x", "40x"]:
#         magnification_path = os.path.join(root_folder, magnification_level)
#         if not os.path.exists(magnification_path):
#             continue

#         # Iterate over FA and PT directories
#         for category in ["FA", "PT"]:
#             category_path = os.path.join(magnification_path, category)
#             if not os.path.exists(category_path):
#                 continue

#             # Iterate over slide folders (e.g., "FA 56B", "PT 41 B")
#             for slide_name in os.listdir(category_path):
#                 slide_folder = os.path.join(category_path, slide_name)
#                 if not os.path.isdir(slide_folder):  # Skip non-folder files
#                     continue

#                 patch_map_csv = os.path.join(slide_folder, "patch_map.csv")
#                 if os.path.exists(patch_map_csv):
#                     print(f"Processing slide: {slide_name} ({magnification_level}, {category})")
#                     reconstruct_image_with_predictions(slide_folder, patch_map_csv, predictions_df, patch_size)

# # Example usage:
# root_folder = r"C:\Users\Vivian\Documents\CONCH\patches"
# predictions_csv = r'C:\Users\Vivian\Documents\CONCH\patch_predictions-ep5.csv'

# process_all_slides(root_folder, predictions_csv)

# ----------------------------------------------
"""
Reconstructing slides with overlayed patch predictions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

# def reconstruct_image_with_predictions(slide_folder, patch_map_csv, predictions_df, patch_size=(224, 224)):
#     """
#     Reconstructs an image from patches and applies green overlays for correct predictions
#     and red overlays for incorrect predictions.
#     """

#     # Load patch map for the specific slide
#     patch_map_df = pd.read_csv(patch_map_csv)

#     # Extract slide name (e.g., "FA 56B" or "PT 41 B") from folder path
#     slide_name = os.path.basename(slide_folder)

#     # Filter predictions only for this slide
#     slide_predictions = predictions_df[predictions_df['Patch Path'].str.contains(slide_name, regex=False)]

#     if slide_predictions.empty:
#         print(f"Skipping {slide_name}: No predictions found.")
#         return

#     # Convert predictions into a dictionary for fast lookup
#     predictions_dict = {os.path.basename(row['Patch Path']): (row['Predicted'], row['True Label'])
#                         for _, row in slide_predictions.iterrows()}

#     # Determine reconstructed image dimensions
#     max_x = patch_map_df['x'].max() + patch_size[0]
#     max_y = patch_map_df['y'].max() + patch_size[1]

#     reconstructed_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)

#     # Initialize overlay positions
#     overlay_boxes = []

#     for _, row in patch_map_df.iterrows():
#         patch_file, x, y = row['patch_file'], row['x'], row['y']

#         # Load the patch
#         patch_path = os.path.join(slide_folder, patch_file)
#         if not os.path.exists(patch_path):
#             continue  # Skip missing files
#         patch = np.load(patch_path)

#         # Ensure patches are RGB
#         if patch.ndim == 2:  # Convert grayscale to RGB
#             patch = np.stack([patch]*3, axis=-1)

#         # Add patch to the reconstructed image
#         reconstructed_image[y:y + patch_size[1], x:x + patch_size[0], :] = patch

#         # Check if the patch has a prediction
#         if patch_file in predictions_dict:
#             predicted, true_label = predictions_dict[patch_file]
#             overlay_color = 'green' if predicted == true_label else 'red'
#             overlay_boxes.append((x, y, overlay_color))

#     # Display the reconstructed image
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(reconstructed_image)
#     ax.axis('off')
#     ax.set_title(f"Reconstructed Slide: {slide_name}", fontsize=16)

#     # Draw overlay boxes
#     for (x, y, color) in overlay_boxes:
#         rect = Rectangle((x, y), patch_size[0], patch_size[1], linewidth=2, edgecolor=color, facecolor='none')
#         ax.add_patch(rect)

#     plt.show()

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

# Example usage:
root_folder = r"C:\Users\Vivian\Documents\CONCH\patches_annotated"
# predictions_csv = r'C:\Users\Vivian\Documents\CONCH\patch_predictions\patient_split_UNI70_linprob.csv'
predictions_csv = r'C:\Users\Vivian\Documents\CONCH\patch_predictions\UNI_linprob_test_ann_only.csv'

process_selected_slides(root_folder, predictions_csv)

# # ----------------------------------------------
# # overlaying translucent boxes instead of outlines
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# def reconstruct_image_with_predictions(slide_folder, patch_map_csv, predictions_df, patch_size=(224, 224), alpha=0.4):
#     """
#     Reconstructs an image from patches and applies translucent green overlays for correct predictions
#     and translucent red overlays for incorrect predictions.
#     """

#     # Load patch map for the specific slide
#     patch_map_df = pd.read_csv(patch_map_csv)

#     # Extract slide name (e.g., "FA 56B" or "PT 41 B") from folder path
#     slide_name = os.path.basename(slide_folder)

#     # Filter predictions only for this slide
#     slide_predictions = predictions_df[predictions_df['Patch Path'].str.contains(slide_name, regex=False)]

#     if slide_predictions.empty:
#         print(f"Skipping {slide_name}: No predictions found.")
#         return

#     # Convert predictions into a dictionary for fast lookup
#     predictions_dict = {os.path.basename(row['Patch Path']): (row['Predicted'], row['True Label'])
#                         for _, row in slide_predictions.iterrows()}

#     # Determine reconstructed image dimensions
#     max_x = patch_map_df['x'].max() + patch_size[0]
#     max_y = patch_map_df['y'].max() + patch_size[1]

#     reconstructed_image = np.zeros((max_y, max_x, 3), dtype=np.uint8)

#     # Create an overlay mask with same dimensions
#     overlay = np.zeros((max_y, max_x, 3), dtype=np.uint8)

#     for _, row in patch_map_df.iterrows():
#         patch_file, x, y = row['patch_file'], row['x'], row['y']

#         # Load the patch
#         patch_path = os.path.join(slide_folder, patch_file)
#         if not os.path.exists(patch_path):
#             continue  # Skip missing files
#         patch = np.load(patch_path)

#         # Ensure patches are RGB
#         if patch.ndim == 2:  # Convert grayscale to RGB
#             patch = np.stack([patch]*3, axis=-1)

#         # Add patch to the reconstructed image
#         reconstructed_image[y:y + patch_size[1], x:x + patch_size[0], :] = patch

#         # Apply overlay based on predictions
#         if patch_file in predictions_dict:
#             predicted, true_label = predictions_dict[patch_file]
#             color = [0, 255, 0] if predicted == true_label else [255, 0, 0]  # Green for correct, Red for incorrect
#             overlay[y:y + patch_size[1], x:x + patch_size[0], :] = color  # Apply color to overlay mask

#     # Blend overlay with the reconstructed image
#     blended_image = ((1 - alpha) * reconstructed_image + alpha * overlay).astype(np.uint8)

#     # Display the reconstructed image
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.imshow(blended_image)
#     ax.axis('off')
#     ax.set_title(f"Reconstructed Slide: {slide_name}", fontsize=16)

#     plt.show()


# def process_selected_slides(root_folder, predictions_csv, patch_size=(224, 224)):
#     """
#     Processes only the slides found in the predictions CSV.
#     """

#     # Load full predictions CSV
#     predictions_df = pd.read_csv(predictions_csv)

#     # Extract unique slide names from predictions
#     unique_slides = set(predictions_df['Patch Path'].apply(lambda x: os.path.basename(os.path.dirname(x))))

#     # Iterate over magnification levels (20x, 40x)
#     for magnification_level in ["20x", "40x"]:
#         magnification_path = os.path.join(root_folder, magnification_level)
#         if not os.path.exists(magnification_path):
#             continue

#         # Iterate over FA and PT directories
#         for category in ["FA", "PT"]:
#             category_path = os.path.join(magnification_path, category)
#             if not os.path.exists(category_path):
#                 continue

#             # Iterate over slide folders and process only those in the predictions CSV
#             for slide_name in os.listdir(category_path):
#                 slide_folder = os.path.join(category_path, slide_name)
#                 if not os.path.isdir(slide_folder):  # Skip non-folder files
#                     continue

#                 if slide_name not in unique_slides:
#                     continue  # Skip slides not in the predictions CSV

#                 patch_map_csv = os.path.join(slide_folder, "patch_map.csv")
#                 if os.path.exists(patch_map_csv):
#                     print(f"Processing slide: {slide_name} ({magnification_level}, {category})")
#                     reconstruct_image_with_predictions(slide_folder, patch_map_csv, predictions_df, patch_size)

# # Example usage:
# root_folder = r"C:\Users\Vivian\Documents\CONCH\patches"
# predictions_csv = r'C:\Users\Vivian\Documents\CONCH\patch_predictions_UNI.csv'

# process_selected_slides(root_folder, predictions_csv)
