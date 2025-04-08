"""
This script contains functions to:
    1. Extracts metadata from vsi files 
    2. Filters out slides based on sample type (A, B, C) and saves them in a new folder.
    3. Reads the metadata and creates plots to display the statistics. Places FA and PT on x-axis.
"""

import pandas as pd
import ast
import matplotlib.pyplot as plt
import sys
import re
import numpy as np
import pandas as pd
import ast
import os
import os
import subprocess
import pandas as pd
import re
from tqdm import tqdm


# -------------------------------
# 1. Extract metadata from VSI files
# -------------------------------

def extract_metadata(vsi_file, bftools_path):
    """
    Extracts metadata from a single VSI file using Bio-Formats showinf.bat tool.
    """
    cmd = f'"{bftools_path}" -nopix -omexml "{vsi_file}" | findstr "PhysicalSize SizeX SizeY Magnification"'
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   shell=True, universal_newlines=True)
        stdout, stderr = process.communicate()

        if stderr:
            print(f"Error processing {vsi_file}: {stderr}")
            return None

        metadata = {"Filename": os.path.basename(vsi_file)}

        magnifications = re.findall(r'NominalMagnification="([\d\.]+)"', stdout)
        metadata["Magnifications"] = tuple(map(float, magnifications)) if magnifications else "Unknown"

        pixel_sizes = re.findall(r'PhysicalSizeX="([\d\.]+)".*?PhysicalSizeY="([\d\.]+)"',
                                 stdout, re.DOTALL)
        metadata["Pixel Sizes (X, Y)"] = [tuple(map(float, size)) for size in pixel_sizes] if pixel_sizes else "Unknown"

        resolution_matches = re.findall(r'SizeX="(\d+)" SizeY="(\d+)"', stdout)
        metadata["Resolution Levels"] = [tuple(map(int, res)) for res in resolution_matches] if resolution_matches else "Unknown"

        metadata["Number of Levels"] = len(metadata["Resolution Levels"]) if isinstance(metadata["Resolution Levels"], list) else 0

        return metadata

    except subprocess.CalledProcessError:
        print(f"Error reading {vsi_file}")
        return None

def extract_metadata_from_folder(bftools_path, vsi_folder, output_csv):
    """
    Extracts metadata from all VSI files in a given folder and saves it to a CSV file.

    Parameters:
        bftools_path (str): Path to Bio-Formats showinf.bat tool.
        vsi_folder (str): Directory containing VSI files.
        output_csv (str): Path to save the output metadata CSV.
    """
    vsi_files = [os.path.join(vsi_folder, f) for f in os.listdir(vsi_folder) if f.endswith(".vsi")]

    metadata_list = []
    for vsi_file in tqdm(vsi_files, desc="Processing VSI files", unit="file"):
        metadata = extract_metadata(vsi_file, bftools_path)
        if metadata:
            metadata_list.append(metadata)

    df = pd.DataFrame(metadata_list)
    df.to_csv(output_csv, index=False)
    print(f"Metadata saved to {output_csv}")

# ------------------------------------------------------------------------------
# 2. Filter samples by type and save them
# ------------------------------------------------------------------------------

def filter_and_save_samples_by_type(csv_files, sample_type_letter, output_dir="metadata/filtered_slides"):
    """
    Filters the metadata CSV files for each class (PT and FA) to only include samples of the specified type
    (e.g., "A", "B", "C") and saves separate CSV files for each magnification level within each class.

    Parameters:
        csv_files (dict): Dictionary with class names as keys and paths to CSV files as values.
        sample_type_letter (str): The letter indicating the tissue type to filter (e.g., "B").
        output_dir (str): Directory to save the filtered output CSVs.
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name, csv_path in csv_files.items():
        metadata_df = pd.read_csv(csv_path)
        metadata_df['Magnifications'] = metadata_df['Magnifications'].apply(ast.literal_eval)
        metadata_df['Max_Magnification'] = metadata_df['Magnifications'].apply(max)

        # Filter samples by specified type letter (e.g., "B")
        filtered_samples = metadata_df[metadata_df['Filename'].str.contains(sample_type_letter)]

        # Save separate CSVs for each magnification level within each class
        for mag, group in filtered_samples.groupby('Max_Magnification'):
            output_csv = os.path.join(
                output_dir, f"{class_name}_filtered_slides_{sample_type_letter}_{int(mag)}x.csv"
            )
            group.to_csv(output_csv, index=False)
            print(f"Saved {class_name} slides for type {sample_type_letter} at {int(mag)}x magnification: {output_csv}")


# ------------------------------------------------------------
# 3. Load CSV and extract metadata for visualization
# ------------------------------------------------------------

def load_csv(file_path):
    """Load the CSV file into a pandas DataFrame."""
    df = pd.read_csv(file_path)

    # Convert string representations to lists/tuples
    df["Magnifications"] = df["Magnifications"].apply(ast.literal_eval)
    df["Pixel Sizes (X, Y)"] = df["Pixel Sizes (X, Y)"].apply(ast.literal_eval)

    return df

def extract_metadata(df, lesion_class):
    """Extract relevant metadata for visualization."""
    # Extract minimum pixel size (rounded to 2 decimals)
    df["Min Pixel Size"] = df["Pixel Sizes (X, Y)"].apply(lambda x: round(min([val[0] for val in x]), 2))

    def get_sample_type(filename):
        # Use regex to extract the last character before ".vsi"
        match = re.search(r'([A-C])\d*\.vsi$', filename)
        if match:
            sample_type = match.group(1)
            if sample_type == "A":
                return "Biopsy (A)"
            elif sample_type == "B":
                return "Tissue (B)"
            elif sample_type == "C":
                return "Other (C)"
        return "Other"  # Default case if no match is found

    df["Sample Type"] = df["Filename"].apply(get_sample_type)

    # Extract max magnification factor
    df["Max Magnification"] = df["Magnifications"].apply(max)

    # Assign lesion class label
    df["Lesion Class"] = lesion_class

    return df

def plot_stacked_bar_chart(df1, df2, column, x_label, y_label, title, lesion_class1, lesion_class2, colors):
    """Create stacked bar chart comparing FA and PT datasets."""
    plt.figure(figsize=(10, 6))

    # Get unique categories
    unique_categories = sorted(set(df1[column].unique()).union(set(df2[column].unique())))

    # Create counts for FA and PT per category
    counts_fa = df1[column].value_counts().reindex(unique_categories, fill_value=0)
    counts_pt = df2[column].value_counts().reindex(unique_categories, fill_value=0)

    # Define positions for bars
    x_labels = [lesion_class1, lesion_class2]
    x_positions = np.arange(len(x_labels))

    # Initialize bottom positions for stacking
    bottom_fa = np.zeros(len(x_labels))
    bottom_pt = np.zeros(len(x_labels))

    # Iterate through each category to stack bars
    for i, category in enumerate(unique_categories):
        plt.bar(x_positions[0], counts_fa[category], color=colors[i], label=category if x_positions[0] == 0 else "", bottom=bottom_fa[0])
        plt.bar(x_positions[1], counts_pt[category], color=colors[i], bottom=bottom_pt[1])

        # Add counts on bars
        if counts_fa[category] > 0:
            plt.text(x_positions[0], bottom_fa[0] + counts_fa[category] / 2, int(counts_fa[category]), ha="center", va="center", fontsize=12)
        if counts_pt[category] > 0:
            plt.text(x_positions[1], bottom_pt[1] + counts_pt[category] / 2, int(counts_pt[category]), ha="center", va="center", fontsize=12)

        # Update bottom positions for next stacked segment
        bottom_fa[0] += counts_fa[category]
        bottom_pt[1] += counts_pt[category]

    plt.xticks(x_positions, x_labels)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(title=column)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

def generate_plots(df1, df2, lesion_class1, lesion_class2):
    """Generate all required grouped bar plots with correct labels and colors."""

    # Define distinct colors for different categories
    magnification_colors = ["royalblue", "tomato", "gold", "mediumseagreen", "purple", "cyan"]
    sample_type_colors = ["cornflowerblue", "lightcoral", "mediumaquamarine"]

    # Stacked bar chart for Minimum Pixel Size Distribution
    plot_stacked_bar_chart(df1, df2, "Min Pixel Size", "Lesion Class", "Number of Slides",
                           "Distribution of Minimum Pixel Sizes", lesion_class1, lesion_class2, magnification_colors)

    # Stacked bar chart for Sample Type Distribution
    plot_stacked_bar_chart(df1, df2, "Sample Type", "Lesion Class", "Number of Slides",
                           "Distribution of Sample Types", lesion_class1, lesion_class2, sample_type_colors)

    # Stacked bar chart for Max Magnification Factor
    plot_stacked_bar_chart(df1, df2, "Max Magnification", "Lesion Class", "Number of Slides",
                           "Distribution of Max Magnification Factors", lesion_class1, lesion_class2, magnification_colors)

# ------------------------------------------------------------------------------


def main():
    # ---- extracting metadata from vsi files ----
    bftools_path = r"C:\Users\Vivian\Downloads\bftools\bftools\showinf.bat"
    vsi_folder = r"Z:\mirage\med-i_data\Data\Amoon\Pathology Raw\PT scans"
    output_csv = "vsi_PT_metadata.csv"

    extract_metadata_from_folder(bftools_path, vsi_folder, output_csv)


    # ---- creating plots ----

    # # File paths for FA and PT lesion class CSV files
    # file_path_1 = r"C:\Users\Vivian\Documents\CONCH\vsi_FA_metadata.csv"
    # file_path_2 = r"C:\Users\Vivian\Documents\CONCH\vsi_PT_metadata.csv"

    # lesion_class1, lesion_class2 = "FA", "PT"
    # df1 = load_csv(file_path_1)
    # df2 = load_csv(file_path_2)

    # df1 = extract_metadata(df1, lesion_class1)
    # df2 = extract_metadata(df2, lesion_class2)

    # generate_plots(df1, df2, lesion_class1, lesion_class2)

    # ---------filter by sample type ---------
    # csv_files = {
    #     "PT": r"C:\Users\Vivian\Documents\CONCH\vsi_PT_metadata.csv",
    #     "FA": r"C:\Users\Vivian\Documents\CONCH\vsi_FA_metadata.csv"
    # }
    # sample_type = "B"  # Change this to "A" or "C" as needed
    # filter_and_save_samples_by_type(csv_files, sample_type)



if __name__ == "__main__":
    main()
