# """
# This program reads the metadata and creates plots to display the statistics.
# """
# import pandas as pd
# import ast
# import matplotlib.pyplot as plt
# import sys
# import re

# def load_csv(file_path):
#     """Load the CSV file into a pandas DataFrame."""
#     df = pd.read_csv(file_path)

#     # Convert string representations to lists/tuples
#     df["Magnifications"] = df["Magnifications"].apply(ast.literal_eval)
#     df["Pixel Sizes (X, Y)"] = df["Pixel Sizes (X, Y)"].apply(ast.literal_eval)

#     return df

# def extract_metadata(df, lesion_class):
#     """Extract relevant metadata for visualization."""
#     # Extract minimum pixel size (rounded to 2 decimals)
#     df["Min Pixel Size"] = df["Pixel Sizes (X, Y)"].apply(lambda x: round(min([val[0] for val in x]), 2))

#     def get_sample_type(filename):
#         # Use regex to extract the last character before ".vsi"
#         match = re.search(r'([A-C])\d*\.vsi$', filename)
#         if match:
#             sample_type = match.group(1)
#             if sample_type == "A":
#                 return "Biopsy"
#             elif sample_type == "B":
#                 return "Tissue"
#             elif sample_type == "C":
#                 return "Other"
#         return "Other"  # Default case if no match is found

#     df["Sample Type"] = df["Filename"].apply(get_sample_type)

#     # Extract max magnification factor
#     df["Max Magnification"] = df["Magnifications"].apply(max)

#     # Assign lesion class label
#     df["Lesion Class"] = lesion_class

#     return df

# def plot_grouped_bar_chart(df1, df2, column, x_label, y_label, title, lesion_class1, lesion_class2, colors):
#     """Create grouped bar chart comparing two datasets with labeled bars."""
#     plt.figure(figsize=(10, 6))

#     # Count values for each dataset
#     counts_df1 = df1[column].value_counts().sort_index()
#     counts_df2 = df2[column].value_counts().sort_index()

#     # Align indexes (fill missing values with 0)
#     all_indexes = sorted(set(counts_df1.index).union(set(counts_df2.index)))
#     counts_df1 = counts_df1.reindex(all_indexes, fill_value=0)
#     counts_df2 = counts_df2.reindex(all_indexes, fill_value=0)

#     bar_width = 0.4
#     x = range(len(all_indexes))

#     bars1 = plt.bar(x, counts_df1, width=bar_width, label=lesion_class1, color=colors[0], alpha=0.7)
#     bars2 = plt.bar([p + bar_width for p in x], counts_df2, width=bar_width, label=lesion_class2, color=colors[1], alpha=0.7)

#     # Add counts on top of bars
#     for bars in [bars1, bars2]:
#         for bar in bars:
#             height = bar.get_height()
#             if height > 0:
#                 plt.text(bar.get_x() + bar.get_width()/2, height, int(height), ha="center", va="bottom", fontsize=12)

#     plt.xticks([p + bar_width / 2 for p in x], all_indexes, rotation=45)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
#     plt.legend()
#     plt.grid(axis="y", linestyle="--", alpha=0.7)
#     plt.show()

# def generate_plots(df1, df2, lesion_class1, lesion_class2):
#     """Generate all required grouped bar plots with correct labels and colors."""
#     # Define colors for FA (blue) and PT (red)
#     lesion_colors = ["mediumaquamarine", "lightcoral"]
#     # lesion_colors = ["cornflowerblue", "lightcoral"]

#     # Grouped bar plot for Minimum Pixel Size Distribution
#     plot_grouped_bar_chart(df1, df2, "Min Pixel Size", "Minimum Pixel Size", "Number of Slides",
#                            "Distribution of Minimum Pixel Sizes", lesion_class1, lesion_class2, lesion_colors)

#     # Define colors for sample types
#     # sample_type_colors = ["blue", "red", "green"]  # Biopsy (A) = Blue, Tissue (B) = Red, Other = Green
#     plot_grouped_bar_chart(df1, df2, "Sample Type", "Sample Type", "Number of Slides",
#                            "Distribution of Sample Types", lesion_class1, lesion_class2, lesion_colors)

#     # Grouped bar plot for Max Magnification Factor
#     plot_grouped_bar_chart(df1, df2, "Max Magnification", "Max Magnification Factor", "Number of Slides",
#                            "Distribution of Max Magnification Factors", lesion_class1, lesion_class2, lesion_colors)

# def main():
#     # File paths for FA and PT lesion class CSV files
#     file_path_1 = r"C:\Users\Vivian\Documents\CONCH\vsi_FA_metadata.csv"
#     file_path_2 = r"C:\Users\Vivian\Documents\CONCH\vsi_PT_metadata.csv"

#     lesion_class1, lesion_class2 = "FA", "PT"
#     df1 = load_csv(file_path_1)
#     df2 = load_csv(file_path_2)

#     df1 = extract_metadata(df1, lesion_class1)
#     df2 = extract_metadata(df2, lesion_class2)

#     generate_plots(df1, df2, lesion_class1, lesion_class2)

# if __name__ == "__main__":
#     main()

# FA and PT on x-axis
"""
This program reads the metadata and creates plots to display the statistics.
"""

import pandas as pd
import ast
import matplotlib.pyplot as plt
import sys
import re
import numpy as np

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

def main():
    # File paths for FA and PT lesion class CSV files
    file_path_1 = r"C:\Users\Vivian\Documents\CONCH\vsi_FA_metadata.csv"
    file_path_2 = r"C:\Users\Vivian\Documents\CONCH\vsi_PT_metadata.csv"

    lesion_class1, lesion_class2 = "FA", "PT"
    df1 = load_csv(file_path_1)
    df2 = load_csv(file_path_2)

    df1 = extract_metadata(df1, lesion_class1)
    df2 = extract_metadata(df2, lesion_class2)

    generate_plots(df1, df2, lesion_class1, lesion_class2)

if __name__ == "__main__":
    main()
