# filter out patches that are not in the mask
import os
import pandas as pd
from tqdm import tqdm

def filter_patches_by_mask(test_patches_path, patch_mask_root, output_filtered_csv):
    """
    Filters a CSV of test patches using patch-level masks stored in separate CSVs.

    Parameters:
        test_patches_path (str): Path to the input CSV with all patches.
        patch_mask_root (str): Directory containing *_patch_mask.csv files.
        output_filtered_csv (str): Path to save the filtered output CSV.
    """
    # Load test patches
    df = pd.read_csv(test_patches_path)

    # Extract patch file and slide name
    df['patch_file'] = df['image'].apply(lambda x: os.path.basename(x))
    df['slide_name'] = df['image'].apply(lambda x: os.path.basename(os.path.dirname(x)))

    # Load all patch masks into a dictionary
    mask_lookup = {}
    for fname in os.listdir(patch_mask_root):
        if not fname.endswith("_patch_mask.csv"):
            continue
        slide_name = fname.replace("_patch_mask.csv", "")
        df_mask = pd.read_csv(os.path.join(patch_mask_root, fname))
        mask_lookup[slide_name] = df_mask.set_index('patch_file')['InsideAnnotation'].to_dict()

    # Filter patches
    keep_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Filtering patches"):
        patch_file = row['patch_file']
        slide_name = row['slide_name']
        if slide_name in mask_lookup:
            mask = mask_lookup[slide_name]
            if mask.get(patch_file, False):
                keep_rows.append(row)
        else:
            print(f"⚠️ No mask found for slide: {slide_name}")

    # Save filtered DataFrame
    filtered_df = pd.DataFrame(keep_rows)
    filtered_df.to_csv(output_filtered_csv, index=False)
    print(f"✅ Filtered CSV saved to: {output_filtered_csv}")

def main():
    filter_patches_by_mask(
    test_patches_path=r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_5x\test_patches.csv",
    patch_mask_root=r"C:\Users\Vivian\Documents\CONCH\series8_5x_masks",
    output_filtered_csv=r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_5x\test_patches_ann_filtered.csv"
)

if __name__ == "__main__":
    main()