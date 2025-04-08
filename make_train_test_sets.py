# """
# This script generates a train-test split of the slides that have been successfully processed.
# """
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Define patch directory root for both magnifications
# PATCHES_ROOT = "patches"
# MAGNIFICATIONS = [20, 40]  # Check both 20x and 40x

# # List to store successfully processed slides
# successful_slides = []

# # Loop through both magnifications
# for mag in MAGNIFICATIONS:
#     for class_name in ["FA", "PT"]:
#         class_path = os.path.join(PATCHES_ROOT, f"{mag}x", class_name)
        
#         if not os.path.exists(class_path):
#             print(f"Warning: Class directory not found: {class_path}")
#             continue

#         # List all slides inside the class directory
#         slides = [slide for slide in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, slide))]

#         for slide in slides:
#             slide_path = os.path.join(class_path, slide)
#             # Check if slide directory contains patches (i.e., was successfully processed)
#             if len(os.listdir(slide_path)) > 0:
#                 successful_slides.append({"Filename": slide, "Class": class_name, "Magnification": mag})

# # Convert to DataFrame for easier handling
# slides_df = pd.DataFrame(successful_slides)

# # Save successfully processed slides for reference
# # slides_df.to_csv("metadata/successful_slides.csv", index=False)
# # print(f"Found {len(slides_df)} successfully processed slides.")

# # Ensure we have enough slides before proceeding
# if len(slides_df) < 13:
#     print("Not enough successfully processed slides. Try processing more slides before running training.")
#     exit()

# # Split into Train (10) & Test (3) Sets with Stratification
# train_slides, test_slides = train_test_split(
#     slides_df, 
#     test_size=65,
#     train_size=150,
#     stratify=slides_df[["Class", "Magnification"]],  # Ensure balance
#     random_state=42
# )

# # Save to CSV for reference
# train_slides.to_csv("metadata/train_all_slides.csv", index=False)
# test_slides.to_csv("metadata/test_all_slides.csv", index=False)

# print(f"Train set: {len(train_slides)} slides")
# print(f"Test set: {len(test_slides)} slides")

# ----------------------------------------------
# """
# Create 90/10 split
# """
# import pandas as pd
# import random

# # File paths (Modify these paths as needed)
# train_csv_path = r"C:\Users\Vivian\Documents\CONCH\metadata\train_all_slides.csv"
# test_csv_path = r"C:\Users\Vivian\Documents\CONCH\metadata\test_all_slides.csv"

# # Load CSVs
# train_df = pd.read_csv(train_csv_path)
# test_df = pd.read_csv(test_csv_path)

# # Ensure column names are consistent (modify if needed)
# assert "Filename" in train_df.columns and "Filename" in test_df.columns, "Column names mismatch."

# # Slides to **keep** in the test set
# slides_to_keep = {"FA 74 B", "FA 56B", "PT 41 B"}

# # Filter the test set: Keep only the specified slides
# test_df_filtered = test_df[test_df["Filename"].isin(slides_to_keep)]

# # Move all **remaining** test slides to the train set
# moved_to_train = test_df[~test_df["Filename"].isin(slides_to_keep)]
# train_df = pd.concat([train_df, moved_to_train], ignore_index=True)

# # **Recalculate the split**: Ensure 10% of slides remain in test
# num_test_slides = max(1, int(0.1 * len(train_df)))  # At least 1 test sample

# # Randomly select additional test slides (excluding the ones we already kept)
# remaining_train_slides = train_df[~train_df["Filename"].isin(slides_to_keep)]
# new_test_samples = remaining_train_slides.sample(n=num_test_slides - len(test_df_filtered), random_state=42)

# # Update train and test sets
# train_df = train_df[~train_df["Filename"].isin(new_test_samples["Filename"])]
# test_df = pd.concat([test_df_filtered, new_test_samples], ignore_index=True)

# # Save the new CSVs
# train_df.to_csv("metadata\\train_90.csv", index=False)
# test_df.to_csv("metadata\\test_10.csv", index=False)

# print("New train and test CSVs generated successfully!")
# print(f"New train set size: {len(train_df)}")
# print(f"New test set size: {len(test_df)}")

# ----------------------------------------------
"""
attempting to split data by patient number and keeping annotated slides in test set
"""
# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split

# # Load metadata
# df = pd.read_csv("metadata/successful_slides.csv")

# # Normalize filenames by removing spaces and converting to uppercase
# def normalize(name):
#     return name.replace(" ", "").upper()

# df["NormalizedFilename"] = df["Filename"].apply(normalize)

# # Annotated filenames (manually listed)
# annotated_filenames_raw = [
#     # FA
#     "FA 56B", "FA 57B", "FA 58B", "FA 59B", "FA 60 B", "FA 61 B", "FA 62 B",
#     "FA 63 B", "FA 64 B", "FA 65 B", "FA 66 B", "FA 67 B", "FA 68 B", "FA 70 B",
#     "FA 71 B", "FA 73 B", "FA 74 B", "FA 75 B", "FA 76 B", "FA 77 B", "FA 78 B",
#     "FA 85 B", "FA 86 B",
#     # PT
#     "PT 35 B", "PT 36 B", "PT 37 B", "PT 39 B", "PT 40 B", "PT 41 B", "PT 42 B", "PT 52 B"
# ]
# annotated_filenames_norm = [normalize(name) for name in annotated_filenames_raw]

# # Select annotated slides by normalized match
# test_df = df[df["NormalizedFilename"].isin(annotated_filenames_norm)].copy()
# train_candidates = df[~df.index.isin(test_df.index)].copy()

# # Define desired test set size (e.g., 20%)
# desired_test_size = 0.2
# remaining_test_size = int(desired_test_size * len(df)) - len(test_df)
# remaining_test_size = max(0, remaining_test_size)

# # Random sample the rest of test set if needed
# if remaining_test_size > 0:
#     additional_test_df = train_candidates.sample(n=remaining_test_size, random_state=42)
#     train_df = train_candidates.drop(index=additional_test_df.index)
#     final_test_df = pd.concat([test_df, additional_test_df])
# else:
#     train_df = train_candidates.copy()
#     final_test_df = test_df.copy()

# # Drop helper column
# train_df = train_df.drop(columns=["NormalizedFilename"])
# final_test_df = final_test_df.drop(columns=["NormalizedFilename"])

# # Save to CSV
# os.makedirs("metadata", exist_ok=True)
# train_df.to_csv("metadata/train_split.csv", index=False)
# final_test_df.to_csv("metadata/test_split.csv", index=False)

# print(f"Train set: {len(train_df)} slides")
# print(f"Test set: {len(final_test_df)} slides (including {len(test_df)} annotated)")

# ----------------------------------------------
"""
creates a train-test(val) split based on patient ID
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the slide metadata
csv_path = r"metadata\patient_split_annotate\slide_csv\train_val_slides.csv"  # Update to your actual path
df = pd.read_csv(csv_path)

# Extract patient ID from the filename (e.g., "FA 100 B1" → "FA_100")
def extract_patient_id(filename):
    parts = filename.split()
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    else:
        return filename  # fallback

df['PatientID'] = df['Filename'].apply(extract_patient_id)

# Drop duplicate PatientID entries, keeping one slide per patient for stratification
patients_df = df.drop_duplicates(subset='PatientID')[['PatientID', 'Class', 'Magnification']]

# Print group sizes to debug stratification
group_sizes = patients_df.groupby(['Class', 'Magnification']).size()
print("Patient counts by (Class, Magnification):")
print(group_sizes)

# Train-test split based on patient, stratifying by class and magnification
train_patients, test_patients = train_test_split(
    patients_df,
    test_size=0.2,
    # stratify=patients_df[['Class', 'Magnification']],
    stratify=patients_df['Class'],
    random_state=42
)

# Now map back to original slides
train_df = df[df['PatientID'].isin(train_patients['PatientID'])]
test_df = df[df['PatientID'].isin(test_patients['PatientID'])]

# Drop PatientID column if not needed in output
train_df = train_df.drop(columns=['PatientID'])
test_df = test_df.drop(columns=['PatientID'])

# Save output CSVs
train_df.to_csv("metadata/patient_split_annotate/patch_csv/train_split.csv", index=False)
test_df.to_csv("metadata/patient_split_annotate/patch_csv/val_split.csv", index=False)

print(f"Train set: {len(train_df)} slides")
print(f"Val set: {len(test_df)} slides")

# ----------------------------------------------
"""
create test set for annotated slides that were manually tiled.
"""

# import os
# import pandas as pd

# def generate_test_slide_csv(root_dir, output_csv, magnification=20):
#     """
#     Generates a CSV file listing slide directories from a given dataset structure.
    
#     Args:
#         root_dir (str): Root directory containing class folders (e.g., FA and PT).
#         output_csv (str): Path where the CSV will be saved.
#         magnification (int or str): Magnification level to assign to all slides.
#     """
#     rows = []

#     # Iterate over classes
#     for class_name in ["FA", "PT"]:
#         class_dir = os.path.join(root_dir, class_name)
#         if not os.path.isdir(class_dir):
#             continue

#         # Iterate over patient/slide directories within each class
#         for slide_folder in sorted(os.listdir(class_dir)):
#             slide_path = os.path.join(class_dir, slide_folder)
#             if os.path.isdir(slide_path):
#                 rows.append({
#                     "Filename": slide_folder,
#                     "Class": class_name,
#                     "Magnification": magnification
#                 })

#     # Create DataFrame and save
#     df = pd.DataFrame(rows)
#     df.to_csv(output_csv, index=False)
#     print(f"Saved test slide metadata to: {output_csv}")

# generate_test_slide_csv(
#     root_dir=r"patches_annotated\20x",
#     output_csv=r"metadata\test_ann_series8.csv"
# )

# ------------------------------------------------

