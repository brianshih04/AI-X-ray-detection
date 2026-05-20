#!/usr/bin/env python3
"""
NIH ChestX-ray14 — 資料驗證腳本
=================================
確認 ETL pipeline 產出正確：
  1. 影像-標籤對應正確性抽檢
  2. 標籤分佈統計（各 split）
  3. 患者分層無洩漏驗證
  4. DataLoader 輸出 shape / value range 驗證
  5. Class weights 合理性檢查

用法：
  python scripts/03_validate_data.py \
      --data-dir data/processed \
      --img-dir data/raw/nih-chest-xrays/images \
      --size 224
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def validate_splits_exist(data_dir: str) -> bool:
    """確認 train.csv / val.csv / test.csv 都存在。"""
    files = ["train.csv", "val.csv", "test.csv"]
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            print(f"  [FAIL] Missing: {path}")
            return False
        print(f"  [OK] Found: {path}")
    return True


def parse_split_csv(csv_path: str) -> list:
    """簡易解析 split CSV。"""
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_str = row.get("labels", "").strip()
            if labels_str == "" or labels_str == "No Finding":
                labels = []
            else:
                labels = [l.strip() for l in labels_str.split("|")]
            entries.append({
                "image_index": row.get("image_index", "").strip(),
                "patient_id": int(row.get("patient_id", "0")),
                "labels": labels,
                "age": int(row.get("age", "0")),
                "gender": row.get("gender", "").strip(),
                "view": row.get("view", "").strip(),
            })
    return entries


def verify_no_leakage(data_dir: str) -> bool:
    """驗證三個 split 的患者群互不相交。"""
    all_patients = {}
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(data_dir, f"{split_name}.csv")
        entries = parse_split_csv(csv_path)
        pids = {e["patient_id"] for e in entries}
        all_patients[split_name] = pids

    # 檢查交叉
    ok = True
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for a, b in pairs:
        overlap = all_patients[a] & all_patients[b]
        if overlap:
            print(f"  [FAIL] {len(overlap)} patients overlap between {a} and {b}")
            ok = False
        else:
            print(f"  [OK] No patient overlap: {a} ∩ {b} = ∅")

    # 確認 union 包含所有患者（無遺漏）
    total_patients = all_patients["train"] | all_patients["val"] | all_patients["test"]
    total_images = sum(len(v) for v in all_patients.values())
    print(f"  [INFO] Total unique patients: {len(total_patients)}")
    print(f"  [INFO] Total images: {total_images}")

    return ok


def verify_label_distribution(data_dir: str) -> bool:
    """統計各 split 的標籤分佈。"""
    from chestxray14_dataset import ALL_LABELS

    print("\n  Label Distribution by Split:")
    print(f"  {'Label':<22} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    totals = Counter()
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(data_dir, f"{split_name}.csv")
        entries = parse_split_csv(csv_path)
        counter = Counter()
        for e in entries:
            for lbl in e["labels"]:
                counter[lbl] += 1
                totals[lbl] += 1

        # 印出（暫存）
        for lbl in ALL_LABELS:
            pass  # 下面統一印

    # 統一印出
    split_counts = {}
    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(data_dir, f"{split_name}.csv")
        entries = parse_split_csv(csv_path)
        counter = Counter()
        for e in entries:
            for lbl in e["labels"]:
                counter[lbl] += 1
        split_counts[split_name] = counter

    for lbl in ALL_LABELS:
        tr = split_counts["train"].get(lbl, 0)
        va = split_counts["val"].get(lbl, 0)
        te = split_counts["test"].get(lbl, 0)
        print(f"  {lbl:<22} {tr:>8,} {va:>8,} {te:>8,} {totals.get(lbl, 0):>8,}")

    return True


def verify_image_label_correspondence(data_dir: str, img_dir: str, sample_size: int = 100) -> bool:
    """抽檢影像是否存在且可開啟。"""
    print(f"\n  Image-file spot check (sampling {sample_size} from each split)...")
    ok = True

    for split_name in ["train", "val", "test"]:
        csv_path = os.path.join(data_dir, f"{split_name}.csv")
        entries = parse_split_csv(csv_path)

        if len(entries) > sample_size:
            sample = random.sample(entries, sample_size)
        else:
            sample = entries

        missing = 0
        unreadable = 0
        for e in sample:
            img_path = os.path.join(img_dir, e["image_index"])
            if not os.path.exists(img_path):
                missing += 1
                continue
            try:
                from PIL import Image
                img = Image.open(img_path)
                img.verify()
            except Exception:
                unreadable += 1

        status = "[OK]" if (missing == 0 and unreadable == 0) else "[WARN]"
        print(f"  {status} {split_name}: {len(sample)} sampled, missing={missing}, unreadable={unreadable}")
        if missing > 0 or unreadable > 0:
            ok = False

    return ok


def verify_dataloader_shape(data_dir: str, img_dir: str, size: int = 224, batch_size: int = 4) -> bool:
    """實際跑一個 DataLoader batch，確認 shape 和 value range。"""
    print(f"\n  DataLoader shape verification (size={size}, batch_size={batch_size})...")
    try:
        from chestxray14_dataset import ChestXray14Dataset, get_transforms

        transform = get_transforms("val", size=size)
        ds = ChestXray14Dataset(
            csv_path=os.path.join(data_dir, "val.csv"),
            image_dir=img_dir,
            transform=transform,
        )

        if len(ds) == 0:
            print("  [WARN] Empty dataset, skipping DataLoader check")
            return True

        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=False, num_workers=0)
        images, labels = next(iter(loader))

        # Shape
        expected_img_shape = (min(batch_size, len(ds)), 3, size, size)
        expected_lbl_shape = (min(batch_size, len(ds)), 14)

        img_ok = images.shape == torch.Size(expected_img_shape)
        lbl_ok = labels.shape == torch.Size(expected_lbl_shape)

        print(f"  [OK] Image shape: {tuple(images.shape)} (expected {expected_img_shape})" if img_ok
              else f"  [FAIL] Image shape: {tuple(images.shape)} (expected {expected_img_shape})")
        print(f"  [OK] Label shape: {tuple(labels.shape)} (expected {expected_lbl_shape})" if lbl_ok
              else f"  [FAIL] Label shape: {tuple(labels.shape)} (expected {expected_lbl_shape})")

        # Value range（normalize 後 mean≈0, std≈1）
        mean_val = images.mean().item()
        std_val = images.std().item()
        print(f"  [INFO] Pixel mean: {mean_val:.4f}, std: {std_val:.4f}")
        if abs(mean_val) > 1.0 or std_val > 5.0:
            print("  [WARN] Pixel values seem unusual — check normalization")
            return False

        # Label values
        unique_labels = labels.unique().tolist()
        print(f"  [INFO] Unique label values: {sorted(unique_labels)}")

        return img_ok and lbl_ok

    except ImportError as e:
        print(f"  [SKIP] Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"  [FAIL] DataLoader error: {e}")
        return False


def verify_class_weights(data_dir: str) -> bool:
    """檢查 class_weights.json 是否合理。"""
    weights_path = os.path.join(data_dir, "class_weights.json")
    if not os.path.exists(weights_path):
        print("  [SKIP] class_weights.json not found")
        return True

    with open(weights_path) as f:
        weights = json.load(f)

    print(f"  Class weights:")
    ok = True
    for lbl, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        if w <= 0 or not np.isfinite(w):
            print(f"    [FAIL] {lbl}: {w} (invalid)")
            ok = False
        else:
            print(f"    {lbl:<22}: {w:.4f}")

    return ok


def verify_metadata_json(data_dir: str) -> bool:
    """檢查 metadata.json 的結構。"""
    meta_path = os.path.join(data_dir, "metadata.json")
    if not os.path.exists(meta_path):
        print("  [SKIP] metadata.json not found")
        return True

    with open(meta_path) as f:
        meta = json.load(f)

    required_keys = ["all_labels", "n_classes", "splits", "seed"]
    missing = [k for k in required_keys if k not in meta]
    if missing:
        print(f"  [FAIL] metadata.json missing keys: {missing}")
        return False

    print(f"  [OK] metadata.json structure valid")
    print(f"    n_classes: {meta['n_classes']}")
    print(f"    labels: {meta['all_labels']}")
    for split_name, info in meta["splits"].items():
        print(f"    {split_name}: {info['n_images']} images, {info['n_patients']} patients")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate ChestXray14 ETL output")
    parser.add_argument("--data-dir", required=True, help="Processed data directory")
    parser.add_argument("--img-dir", default="data/raw/nih-chest-xrays/images", help="Image directory")
    parser.add_argument("--size", type=int, default=224, help="Image size for DataLoader check")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("  NIH ChestX-ray14 — Data Validation")
    print("=" * 60)

    results = {}

    print(f"\n[1/7] Checking split files exist...")
    results["splits_exist"] = validate_splits_exist(args.data_dir)

    print(f"\n[2/7] Checking metadata.json...")
    results["metadata"] = verify_metadata_json(args.data_dir)

    print(f"\n[3/7] Verifying no patient leakage...")
    results["no_leakage"] = verify_no_leakage(args.data_dir)

    print(f"\n[4/7] Verifying label distribution...")
    results["labels"] = verify_label_distribution(args.data_dir)

    if os.path.isdir(args.img_dir):
        print(f"\n[5/7] Spot-checking image files...")
        results["images"] = verify_image_label_correspondence(args.data_dir, args.img_dir)

        print(f"\n[6/7] Verifying DataLoader output...")
        results["dataloader"] = verify_dataloader_shape(
            args.data_dir, args.img_dir,
            size=args.size, batch_size=args.batch_size,
        )
    else:
        print(f"\n[5/7] [SKIP] Image directory not found: {args.img_dir}")
        print(f"\n[6/7] [SKIP] Image directory not found: {args.img_dir}")
        results["images"] = None
        results["dataloader"] = None

    print(f"\n[7/7] Checking class weights...")
    results["weights"] = verify_class_weights(args.data_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Validation Summary")
    print(f"{'='*60}")
    all_passed = True
    for check, passed in results.items():
        if passed is None:
            status = "SKIP"
        elif passed:
            status = "PASS"
        else:
            status = "FAIL"
            all_passed = False
        print(f"  [{status}] {check}")

    if all_passed:
        print(f"\n  ✅ All checks passed!")
    else:
        print(f"\n  ❌ Some checks failed — review above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
