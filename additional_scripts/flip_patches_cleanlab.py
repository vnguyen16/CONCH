import pandas as pd
import os

def flip_labels_on_issues_CL(original_csv, issues_csv, output_csv, preview=True, preview_rows=10):
    # Clean learning label issues
    # Load original and issues CSVs
    df_original = pd.read_csv(original_csv)
    df_issues = pd.read_csv(issues_csv)

    # Extract filename from full path
    df_issues['filename'] = df_issues['path'].apply(lambda p: os.path.basename(p))
    df_issues = df_issues[df_issues['is_label_issue'] == True]

    issue_files = set(df_issues['filename'])

    # Add filename column for matching
    df_original['filename'] = df_original['image'].apply(lambda p: os.path.basename(p))

    # Identify rows to flip
    mask = df_original['filename'].isin(issue_files)
    affected = df_original[mask].copy()
    affected['original_class'] = affected['class']
    affected['class'] = affected['class'].apply(lambda c: 'PT' if c == 'FA' else 'FA')

    if preview:
        print(f"Total patches flagged as label issues: {len(df_issues)}")
        print(f"Number of matching patches found in original CSV: {mask.sum()}")
        print("\n🔍 Preview of label changes:")
        print(affected[['image', 'original_class', 'class']].head(preview_rows))
        print("\n🟡 Preview mode is ON. No file has been saved.")
        return

    # Apply the changes to the original DataFrame
    df_original.loc[mask, 'class'] = affected['class']

    # Drop helper column
    df_original.drop(columns=['filename'], inplace=True)

    # Save to output
    df_original.to_csv(output_csv, index=False)
    print(f"✅ Saved updated CSV with flipped labels to: {output_csv}")


def flip_labels_on_issues2(original_csv, issues_csv, output_csv, preview=True, preview_rows=10):
    # Load original and issues CSVs
    df_original = pd.read_csv(original_csv)
    df_issues = pd.read_csv(issues_csv)

    # Ensure is_label_issue is boolean
    df_issues['is_label_issue'] = df_issues['is_label_issue'].astype(str).str.lower() == 'true'

    # Filter issues to only those with label issues
    df_issues = df_issues[df_issues['is_label_issue']]

    # Merge on full patch path
    merged = df_original.merge(df_issues[['path', 'is_label_issue']], left_on='image', right_on='path', how='left')

    # Mark rows where label should be flipped
    merged['flip'] = merged['is_label_issue'].fillna(False)

    # Backup original labels for preview
    merged['original_class'] = merged['class']

    # Apply label flipping
    merged.loc[merged['flip'] & (merged['class'] == 'FA'), 'class'] = 'PT'
    merged.loc[merged['flip'] & (merged['class'] == 'PT'), 'class'] = 'FA'

    if preview:
        affected = merged[merged['flip']]
        print(f"🔍 Total patches flagged as label issues: {len(df_issues)}")
        print(f"✅ Number of matching patches found in original CSV: {len(affected)}")
        print("\nPreview of label flips:")
        print(affected[['image', 'original_class', 'class']].head(preview_rows))
        print("\n🟡 Preview mode ON — no changes saved.")
        return

    # Save cleaned file
    merged.drop(columns=['path', 'is_label_issue', 'flip', 'original_class'], inplace=True, errors='ignore')
    merged.to_csv(output_csv, index=False)
    print(f"✅ Cleaned CSV saved to: {output_csv}")

def flip_labels_on_issues(original_csv, issues_csv, output_csv, preview=True, preview_rows=10):
    # Load original and issues CSVs
    df_original = pd.read_csv(original_csv)
    df_issues = pd.read_csv(issues_csv)

    # Ensure the is_label_issue column is boolean
    df_issues['is_label_issue'] = df_issues['is_label_issue'].astype(str).str.lower() == 'true'

    # Extract filename from full path in issues CSV
    df_issues['filename'] = df_issues['path'].apply(lambda p: os.path.basename(p))
    df_issues = df_issues[df_issues['is_label_issue']]

    issue_files = set(df_issues['filename'])

    # Extract filename from image path in original CSV
    df_original['filename'] = df_original['image'].apply(lambda p: os.path.basename(p))

    # Identify rows to flip
    mask = df_original['filename'].isin(issue_files)
    affected = df_original[mask].copy()
    affected['original_class'] = affected['class']
    affected['class'] = affected['class'].apply(lambda c: 'PT' if c == 'FA' else 'FA')

    if preview:
        print(f"Total patches flagged as label issues: {len(df_issues)}")
        print(f"Number of matching patches found in original CSV: {mask.sum()}")
        print("\n🔍 Preview of label changes:")
        print(affected[['image', 'original_class', 'class']].head(preview_rows))
        print("\n🟡 Preview mode is ON. No file has been saved.")
        return

    # Apply the label changes to original dataframe
    df_original.loc[mask, 'class'] = affected['class']

    # Drop helper column
    df_original.drop(columns=['filename'], inplace=True)

    # Save to output
    df_original.to_csv(output_csv, index=False)
    print(f"✅ Saved updated CSV with flipped labels to: {output_csv}")


def main():
    flip_labels_on_issues_CL(
        original_csv=r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv\val_patches.csv",
        issues_csv=r"C:\Users\Vivian\Documents\cleanlab\cleanlab_resnet50.csv",
        output_csv=r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv\val_CL_resnet50_patches.csv",
        preview=False  # Set to True to see what would change without saving
    )

#     flip_labels_on_issues(
#     original_csv=r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv\val_patches.csv",
#     issues_csv=r"C:\Users\Vivian\Documents\cleanlab\cleanlab_patch_issues.csv",
#     output_csv=r"C:\Users\Vivian\Documents\CONCH\metadata\patient_split_annotate\patch_csv\val_cleaned_issue_patches.csv",
#     preview=True  # Set to False to actually save the file
# )

if __name__ == "__main__":
    main()