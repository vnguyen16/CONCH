"""
This program filters the metadata CSV files for each class (PT and FA) to only include "B" samples (whole tissue slides) and 
saves separate CSV files for each magnification level within each class.
"""
import pandas as pd
import ast
import os

# Paths to your class-specific CSV files
csv_files = {
    "PT": r"C:\Users\Vivian\Documents\CONCH\vsi_PT_metadata.csv",
    "FA": r"C:\Users\Vivian\Documents\CONCH\vsi_FA_metadata.csv"
}

output_dir = "metadata/filtered_slides"
os.makedirs(output_dir, exist_ok=True)

# Process each CSV file separately (by class)
for class_name, csv_path in csv_files.items():
    metadata_df = pd.read_csv(csv_path)
    metadata_df['Magnifications'] = metadata_df['Magnifications'].apply(ast.literal_eval)
    metadata_df['Max_Magnification'] = metadata_df['Magnifications'].apply(max)

    # Filter "B" samples (whole tissue slides only)
    b_samples = metadata_df[metadata_df['Filename'].str.contains('B')]

    # Save separate CSVs for each magnification level within each class
    for mag, group in b_samples.groupby('Max_Magnification'):
        output_csv = os.path.join(output_dir, f"{class_name}_filtered_slides_{int(mag)}x.csv")
        group.to_csv(output_csv, index=False)
        print(f"Saved {class_name} slides for {int(mag)}x magnification at {output_csv}.")
