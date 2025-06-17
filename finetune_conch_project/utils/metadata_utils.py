# utils/metadata_utils.py

import os
import pandas as pd

def make_breakhis_metadata(fold: int):
    metadata = pd.DataFrame()
    for mode in ['train', 'test']:
        pathname = f'/Users/Vivian/Documents/CONCH/Folds/Fold {fold}/{mode}/'
        images = os.listdir(pathname)
        for image in images:
            if not image.startswith('SOB'):
                continue
            label = image.split('-')[0].replace('SOB_', '')
            class_name, subclass_name = label.split('_')
            metadata = pd.concat([
                metadata, 
                pd.DataFrame({'image': pathname+image, 'fold': fold, 'mode': mode, 'class': class_name, 'subclass': subclass_name}, index=[0])
            ])
        metadata.to_csv(f'/Users/Vivian/Documents/CONCH/Folds/Fold {fold}/{mode}/metadata.csv', index=False)
    return metadata


def make_metadata_from_slide_csv(base_path, slide_csv_path, output_csv_path, magnification_col='Magnification'):
    df = pd.read_csv(slide_csv_path)
    entries = []
    for _, row in df.iterrows():
        slide_name = row["Filename"]
        class_name = row["Class"]
        magnification = f"{row[magnification_col]}x"
        slide_dir = os.path.join(base_path, magnification, class_name, slide_name)
        if not os.path.exists(slide_dir) or len(os.listdir(slide_dir)) == 0:
            print(f"Skipping {slide_name}: no patches found")
            continue
        for patch_file in os.listdir(slide_dir):
            if patch_file.endswith(".npy"):
                entries.append({
                    "image": os.path.join(slide_dir, patch_file),
                    "fold": 1,
                    "mode": os.path.basename(slide_csv_path).split('_')[0],  # crude fallback
                    "class": class_name,
                    "magnification": magnification
                })
    pd.DataFrame(entries).to_csv(output_csv_path, index=False)
    print(f"✅ Saved {output_csv_path} with {len(entries)} entries.")


def make_metadata_from_split_csv(
    annotated_base_path,
    metadata_dir,
    output_dir,
    split_csv_files={"train": "train_split.csv", "val": "val_split.csv", "test": "test_split.csv"},
):
    os.makedirs(output_dir, exist_ok=True)
    for mode, csv_name in split_csv_files.items():
        path = os.path.join(metadata_dir, csv_name)
        output_csv = os.path.join(output_dir, f"{mode}_patches.csv")
        make_metadata_from_slide_csv(annotated_base_path, path, output_csv)
