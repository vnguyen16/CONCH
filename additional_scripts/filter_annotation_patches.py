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


def filter_predictions_by_test_patches(predictions_csv, filtered_test_csv, output_csv):
    """
    Filters predictions based on a list of filtered annotation test patches (can use with saved prediction csvs).
    Parameters:
        predictions_csv (str): Path to the input CSV with predictions.
        filtered_test_csv (str): Path to the CSV with filtered test patches.
        output_csv (str): Path to save the filtered predictions.
    """
    # Load CSVs
    predictions_df = pd.read_csv(predictions_csv)
    filtered_test_df = pd.read_csv(filtered_test_csv)

    # Get list of valid image paths (full paths)
    # valid_patch_paths = set(filtered_test_df['image']) # for new patches (all patches, contains full path)

    # Extract patch filenames from both
    predictions_df['patch_file'] = predictions_df['Patch Path'].apply(lambda x: os.path.basename(x))
    valid_patch_files = set(filtered_test_df['patch_file']) if 'patch_file' in filtered_test_df.columns else \
                        set(filtered_test_df['image'].apply(lambda x: os.path.basename(x)))

    # Filter
    filtered_predictions = predictions_df[predictions_df['patch_file'].isin(valid_patch_files)]

    # Filter predictions
    # filtered_predictions = predictions_df[predictions_df['Patch Path'].isin(valid_patch_paths)]

    # Save filtered predictions
    filtered_predictions.to_csv(output_csv, index=False)
    print(f"✅ Saved filtered predictions to: {output_csv}")

    return predictions_df

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_patch_metrics(predictions_df):
    y_true = predictions_df['True Label']
    y_pred = predictions_df['Predicted']

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average='binary'),
        "Recall": recall_score(y_true, y_pred, average='binary'),
        "F1 Score": f1_score(y_true, y_pred, average='binary')
    }
    return metrics


def main():
#     filter_patches_by_mask(
#     test_patches_path=r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_10x\test_patches.csv",
#     patch_mask_root=r"C:\Users\Vivian\Documents\CONCH\series7_10x_masks",
#     output_filtered_csv=r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_10x\test_patches_ann_filtered.csv"
# )
    filter_predictions_by_test_patches(
        predictions_csv=r"C:\Users\Vivian\Documents\CONCH\patch_predictions\annotated\ResNet50_ann_test.csv",
        filtered_test_csv=r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv_5x\test_patches_ann_filtered.csv",
        output_csv=r"C:\Users\Vivian\Documents\CONCH\patch_predictions\5x\ResNet50_5x_test_ann2.csv"
    )

    # Load predictions
    predictions_df = pd.read_csv(r"C:\Users\Vivian\Documents\CONCH\patch_predictions\5x\ResNet50_5x_test_ann2.csv")
    # Compute metrics
    metrics = compute_patch_metrics(predictions_df)
    print("Patch Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()