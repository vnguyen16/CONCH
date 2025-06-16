# # saving the image paths to a csv file

# import os
# import pandas as pd

# # Define the directory containing the image files
# image_directory = r"patches_png\PT74_B2_level9"  # Replace with your actual directory

# # Define the output CSV file path
# output_csv_path = r"patches_png\PT74.csv"  # Replace with your desired output path

# # Load the existing CSV file (used as a template)
# template_csv_path = r"metadata\fine_tuning\test_metadata_oneslide.csv"  # Replace with your template CSV file
# df_template = pd.read_csv(template_csv_path)

# # Get a list of all image files in the directory (assuming .npy format)
# image_files = [os.path.join(image_directory, f) for f in os.listdir(image_directory) if f.endswith(".png")]

# # Ensure we have at least as many image files as rows in the template
# if len(image_files) < len(df_template):
#     raise ValueError("Not enough image files in the directory to match the CSV rows.")

# # Replace the first column with new image paths
# df_template.iloc[:, 0] = image_files[:len(df_template)]  # Assign only required number of files

# # Save the modified CSV file
# df_template.to_csv(output_csv_path, index=False)

# print(f"New CSV file saved to: {output_csv_path}")

# ----------------------------------------------

# filtering breakhis dataset for fibroadenoma and phyllodes tumor samples

import pandas as pd

# Load CSV file
csv_path = r'C:\Users\Vivian\Documents\CONCH\Folds\Fold 2\test\metadata.csv'
df = pd.read_csv(csv_path)

# Filter rows where subclass is "F" (Fibroadenoma) or "PT" (Phyllodes Tumor)
filtered_df = df[(df['class'] == 'B') & (df['subclass'].isin(['F', 'PT']))]

# Save filtered CSV (optional)
filtered_csv_path = 'C:\\Users\\Vivian\\Documents\\CONCH\\Folds\\Fold 2\\test\\F_PT_metadata.csv'
filtered_df.to_csv(filtered_csv_path, index=False)

print(f"Filtered dataset has {len(filtered_df)} samples.")
