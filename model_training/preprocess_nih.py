"""
Preprocess NIH ChestX-ray14 dataset for training.

Converts Data_Entry_2017.csv into train.csv / val.csv with:
- Path column pointing to actual image files
- Binary label columns for each disease

Usage:
  python preprocess_nih.py --data_dir ~/ai-xray-detection/data --output_dir ~/ai-xray-detection/data
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


NIH_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural_Thickening",
    "Pneumonia", "Pneumothorax",
]

ALL_LABELS = NIH_LABELS + ["No Finding"]


def find_image_path(image_index: str, data_dir: str) -> str:
    """Find actual image path across images_001-012 folders."""
    for i in range(1, 13):
        folder = f"images_{i:03d}/images"
        path = os.path.join(data_dir, folder, image_index)
        if os.path.exists(path):
            return os.path.join(folder, image_index)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir or data_dir

    print(f"Loading Data_Entry_2017.csv from {data_dir}...")
    df = pd.read_csv(os.path.join(data_dir, "Data_Entry_2017.csv"))
    print(f"  Total rows: {len(df)}")

    # Parse multi-label "Finding Labels" into binary columns
    print("Parsing labels...")
    for label in ALL_LABELS:
        df[label] = df["Finding Labels"].apply(
            lambda x: 1.0 if label in str(x).split("|") else 0.0
        )

    # Build image path lookup
    print("Resolving image paths...")
    image_folders = {}
    for i in range(1, 13):
        folder = os.path.join(data_dir, f"images_{i:03d}", "images")
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.endswith(".png"):
                    image_folders[fname] = f"images_{i:03d}/images/{fname}"

    df["Path"] = df["Image Index"].map(image_folders)
    missing = df["Path"].isna().sum()
    if missing > 0:
        print(f"  WARNING: {missing} images not found, dropping them")
        df = df.dropna(subset=["Path"]).reset_index(drop=True)
    print(f"  Mapped {len(df)} images to paths")

    # Split using official train_val_list.txt and test_list.txt
    train_val_list_path = os.path.join(data_dir, "train_val_list.txt")
    test_list_path = os.path.join(data_dir, "test_list.txt")

    train_val_images = set()
    test_images = set()

    if os.path.exists(train_val_list_path):
        with open(train_val_list_path) as f:
            train_val_images = set(line.strip() for line in f)
        print(f"  train_val_list.txt: {len(train_val_images)} images")

    if os.path.exists(test_list_path):
        with open(test_list_path) as f:
            test_images = set(line.strip() for line in f)
        print(f"  test_list.txt: {len(test_images)} images")

    # Split train_val into train (80%) and val (20%) by patient
    if train_val_images:
        train_val_df = df[df["Image Index"].isin(train_val_images)].copy()
        test_df = df[df["Image Index"].isin(test_images)].copy()
    else:
        train_val_df = df.copy()
        test_df = pd.DataFrame()

    # Patient-level split for train/val
    patient_ids = train_val_df["Patient ID"].unique()
    rng = np.random.RandomState(42)
    rng.shuffle(patient_ids)

    n_val = max(1, int(len(patient_ids) * 0.2))
    val_patients = set(patient_ids[:n_val])

    train_df = train_val_df[~train_val_df["Patient ID"].isin(val_patients)].reset_index(drop=True)
    val_df = train_val_df[train_val_df["Patient ID"].isin(val_patients)].reset_index(drop=True)

    print(f"\nSplit results:")
    print(f"  Train: {len(train_df)} images ({train_df['Patient ID'].nunique()} patients)")
    print(f"  Val:   {len(val_df)} images ({val_df['Patient ID'].nunique()} patients)")
    print(f"  Test:  {len(test_df)} images")

    # Save CSVs
    columns_to_save = ["Path"] + ALL_LABELS

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df[columns_to_save].to_csv(train_path, index=False)
    val_df[columns_to_save].to_csv(val_path, index=False)
    if len(test_df) > 0:
        test_df[columns_to_save].to_csv(test_path, index=False)

    print(f"\nSaved:")
    print(f"  {train_path} ({len(train_df)} rows)")
    print(f"  {val_path} ({len(val_df)} rows)")
    if len(test_df) > 0:
        print(f"  {test_path} ({len(test_df)} rows)")

    # Label distribution
    print(f"\nLabel distribution (train):")
    for label in ALL_LABELS:
        count = int(train_df[label].sum())
        pct = count / len(train_df) * 100
        print(f"  {label:25s}: {count:6d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
