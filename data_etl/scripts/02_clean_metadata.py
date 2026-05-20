#!/usr/bin/env python3
"""
NIH ChestX-ray14 — CSV 解析與清洗腳本
=========================================
功能：
  1. 解析 Data_Entry_2017_v2020.csv
  2. 處理缺失標籤（No Finding 標準化）
  3. 去重（同一 patient_id + image_index 保留第一筆）
  4. 病患層級 split（70/10/20，嚴禁 leak）
  5. 輸出 train.csv / val.csv / test.csv

用法：
  python scripts/02_clean_metadata.py \
      --csv data/raw/nih-chest-xrays/Data_Entry_2017_v2020.csv \
      --img-dir data/raw/nih-chest-xrays/images \
      --out-dir data/processed/ \
      --seed 42
"""

import argparse
import csv
import hashlib
import os
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── 14 個官方標籤 ─────────────────────────────────────────────────────────────
ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Consolidation",
    "Edema", "Hernia",
]
LABEL_TO_IDX = {l: i for i, l in enumerate(ALL_LABELS)}
VALID_LABELS = set(ALL_LABELS)

# 已知不可靠或需移除的標籤（來自文獻建議）
UNRELIABLE_LABELS = {"Infiltration"}  # Rajpurkar et al. 2017 建議移除

# 需清理的標籤異體字（normalize 為官方名稱）
LABEL_ALIASES = {
    "Pleural thickening": "Pleural_Thickening",
    "Hernia ": "Hernia",           # 尾部多餘空白
    "Consolidation ": "Consolidation",
}


# ── CSV 解析 ─────────────────────────────────────────────────────────────────
def parse_csv(csv_path: str) -> List[Dict]:
    """解析原始 CSV，回傳干淨的 entries list。"""
    entries = []
    skipped_no_image = 0
    skipped_bad_labels = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            image_index = row.get("Image Index", "").strip()
            if not image_index:
                skipped_no_image += 1
                continue

            raw_labels = row.get("Finding Labels", "No Finding").strip()

            # 解析標籤（用 | 分隔）
            if raw_labels == "No Finding" or raw_labels == "":
                labels: List[str] = []
            else:
                raw_list = raw_labels.split("|")
                labels = []
                for raw_lbl in raw_list:
                    raw_lbl = raw_lbl.strip()

                    # 套用別名表
                    raw_lbl = LABEL_ALIASES.get(raw_lbl, raw_lbl)

                    # 過濾掉不在官方清單的標籤（保留 No Finding）
                    if raw_lbl in VALID_LABELS:
                        labels.append(raw_lbl)
                    else:
                        skipped_bad_labels += 1

            # 若移除 Infiltration 後全空，則為 No Finding
            labels = [l for l in labels if l not in UNRELIABLE_LABELS]

            # 患者年齡（排除不合理值）
            try:
                age = int(row.get("Patient Age", "0"))
                if age < 0 or age > 120:
                    age = 0
            except ValueError:
                age = 0

            # 性別（標準化）
            gender = row.get("Patient Gender", "Unknown").strip()
            if gender not in ("M", "F"):
                gender = "Unknown"

            # 視角（AP/PA）
            view = row.get("View Position", "Unknown").strip()
            if view not in ("AP", "PA"):
                view = "Unknown"

            # Patient ID
            try:
                patient_id = int(row.get("Patient ID", "0"))
            except ValueError:
                patient_id = 0

            entries.append({
                "image_index": image_index,
                "patient_id": patient_id,
                "labels": labels,
                "age": age,
                "gender": gender,
                "view": view,
                "is_no_finding": len(labels) == 0,
            })

    if skipped_no_image:
        warnings.warn(f"Skipped {skipped_no_image} rows with no Image Index")
    if skipped_bad_labels:
        warnings.warn(f"Skipped/cleaned {skipped_bad_labels} non-standard labels")

    return entries


def deduplicate(entries: List[Dict]) -> List[Dict]:
    """
    去重：同一 (patient_id, image_index) 只保留第一筆。
    避免同一影像重複出現在 CSV 中。
    """
    seen = set()
    unique = []
    dupes_removed = 0

    for e in entries:
        key = (e["patient_id"], e["image_index"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
        else:
            dupes_removed += 1

    if dupes_removed:
        warnings.warn(f"Removed {dupes_removed} duplicate entries")
    return unique


# ── Patient-level split ───────────────────────────────────────────────────────
def patient_level_split(
    entries: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Patient-level split（避免同一病患的影像同時出現在 train/test）。
    依 patient_id 分組後再 shuffle。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)
    np.random.seed(seed)

    # 收集每個 patient 的所有 entries index
    patient_to_entries: Dict[int, List[Dict]] = defaultdict(list)
    for e in entries:
        patient_to_entries[e["patient_id"]].append(e)

    patient_ids = list(patient_to_entries.keys())
    random.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = set(patient_ids[:n_train])
    val_patients   = set(patient_ids[n_train:n_train + n_val])
    test_patients  = set(patient_ids[n_train + n_val:])

    train_entries = [e for e in entries if e["patient_id"] in train_patients]
    val_entries   = [e for e in entries if e["patient_id"] in val_patients]
    test_entries  = [e for e in entries if e["patient_id"] in test_patients]

    return train_entries, val_entries, test_entries


# ── Class weight 計算 ────────────────────────────────────────────────────────
def compute_class_weights(
    entries: List[Dict],
    strategy: str = "median_freq",
) -> np.ndarray:
    """
    計算 14 類的加權。Strategies:
      - "inverse":      w_i = 1 / freq_i
      - "median_freq":  w_i = median(freq) / freq_i   (預設)
      - "sqrt":        w_i = 1 / sqrt(freq_i)
      - "balanced":     w_i = n_samples / (n_classes * freq_i)
    """
    counter = Counter()
    for e in entries:
        for lbl in e["labels"]:
            counter[LABEL_TO_IDX[lbl]] += 1

    n_samples = len(entries)
    n_classes = len(ALL_LABELS)

    freqs = np.zeros(n_classes)
    for i in range(n_classes):
        freqs[i] = counter.get(i, 0) / max(n_samples, 1)

    if strategy == "inverse":
        weights = 1.0 / np.maximum(freqs, 1e-6)
    elif strategy == "median_freq":
        median = np.median(freqs)
        weights = median / np.maximum(freqs, 1e-6)
    elif strategy == "sqrt":
        weights = 1.0 / np.sqrt(np.maximum(freqs, 1e-6))
    elif strategy == "balanced":
        weights = n_samples / (n_classes * np.maximum(freqs, 1e-6))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return weights


def verify_no_leakage(
    train: List[Dict],
    val: List[Dict],
    test: List[Dict],
) -> bool:
    """驗證三個 split 的患者群是否互不相交。"""
    train_pids = {e["patient_id"] for e in train}
    val_pids   = {e["patient_id"] for e in val}
    test_pids  = {e["patient_id"] for e in test}

    val_leak  = val_pids & train_pids
    test_leak = test_pids & train_pids

    if val_leak:
        warnings.warn(f"[LEAK] {len(val_leak)} val patients also in train!")
        return False
    if test_leak:
        warnings.warn(f"[LEAK] {len(test_leak)} test patients also in train!")
        return False
    if val_pids & test_pids:
        warnings.warn(f"[LEAK] {len(val_pids & test_pids)} patients in both val and test!")
        return False

    return True


# ── 標籤分佈統計 ──────────────────────────────────────────────────────────────
def label_distribution(entries: List[Dict]) -> Counter:
    counter = Counter()
    for e in entries:
        for lbl in e["labels"]:
            counter[lbl] += 1
    return counter


def print_stats(name: str, entries: List[Dict]):
    n = len(entries)
    n_patients = len({e["patient_id"] for e in entries})
    n_no_find  = sum(1 for e in entries if e["is_no_finding"])
    dist = label_distribution(entries)

    print(f"\n{'='*60}")
    print(f"  [{name}]  {n:,} images  |  {n_patients:,} patients  |  No Finding: {n_no_find} ({n_no_find/n*100:.1f}%)")
    print(f"{'='*60}")
    print(f"  {'Label':<25}  {'Count':>7}  {'Rate':>7}  {'Imbalance':>10}")
    print(f"  {'-'*25}  {'-'*7}  {'-'*7}  {'-'*10}")

    total_labels = sum(dist.values())
    for lbl in ALL_LABELS:
        cnt = dist.get(lbl, 0)
        rate = cnt / n * 100
        # 顯示相對於最高頻類別的倍數
        max_cnt = dist.most_common(1)[0][1] if dist else 1
        imbalance = max_cnt / max(cnt, 1)
        print(f"  {lbl:<25}  {cnt:>7,}  {rate:>6.2f}%  {imbalance:>9.1f}x")


# ── 寫出 CSV ─────────────────────────────────────────────────────────────────
def write_entries_csv(entries: List[Dict], out_path: str):
    """寫出 CSV（含 labels 作為 | 分隔字串）。"""
    fieldnames = ["image_index", "patient_id", "labels", "age", "gender", "view", "is_no_finding"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for e in entries:
            row = dict(e)
            row["labels"] = "|".join(e["labels"]) if e["labels"] else ""
            writer.writerow(row)
    print(f"  -> {out_path}  ({len(entries):,} rows)")


# ── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Clean & split NIH ChestX-ray14 CSV")
    parser.add_argument("--csv", required=True, help="Path to Data_Entry_2017_v2020.csv")
    parser.add_argument("--img-dir", default=None, help="Path to images/ (for optional integrity check)")
    parser.add_argument("--out-dir", default="./data/processed", help="Output directory")
    parser.add_argument("--remove-unreliable", action="store_true",
                        help=f"Remove unreliable labels: {UNRELIABLE_LABELS}")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Parse but don't write files")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n[1/6] Parsing CSV: {args.csv}")
    raw_entries = parse_csv(args.csv)
    print(f"  Raw entries: {len(raw_entries):,}")

    print(f"\n[2/6] Deduplication...")
    entries = deduplicate(raw_entries)
    print(f"  After dedup: {len(entries):,}")

    # 影像存在性抽檢（若 img_dir 提供）
    if args.img_dir and os.path.isdir(args.img_dir):
        print(f"\n[2b/6] Checking image files exist (sample 1000)...")
        missing = 0
        sample = random.sample(entries, min(1000, len(entries)))
        for e in sample:
            if not os.path.exists(os.path.join(args.img_dir, e["image_index"])):
                missing += 1
        print(f"  Missing in sample of {len(sample)}: {missing}")

    print(f"\n[3/6] Patient-level split (70/10/20, seed={args.seed})...")
    train, val, test = patient_level_split(entries, seed=args.seed)

    print_stats("TRAIN", train)
    print_stats("VAL",   val)
    print_stats("TEST",  test)

    print(f"\n[4/6] Verifying no patient leakage...")
    ok = verify_no_leakage(train, val, test)
    if ok:
        print("  [OK] Patient groups are mutually exclusive")
    else:
        print("  [FAIL] LEAKAGE DETECTED — aborting!")
        sys.exit(1)

    if not args.dry_run:
        print(f"\n[5/6] Writing CSV files to {args.out_dir}/...")
        write_entries_csv(train, os.path.join(args.out_dir, "train.csv"))
        write_entries_csv(val,   os.path.join(args.out_dir, "val.csv"))
        write_entries_csv(test,  os.path.join(args.out_dir, "test.csv"))

        # 寫出 class weights (median_freq) for training set
        print(f"\n[6/6] Computing class weights (median_freq)...")
        weights = compute_class_weights(train, strategy="median_freq")
        import json
        weights_path = os.path.join(args.out_dir, "class_weights.json")
        with open(weights_path, "w") as f:
            json.dump({l: float(w) for l, w in zip(ALL_LABELS, weights)}, f, indent=2)
        print(f"  -> {weights_path}")

        # 寫出 splits metadata
        meta = {
            "all_labels": ALL_LABELS,
            "n_classes": len(ALL_LABELS),
            "splits": {
                "train": {"n_images": len(train), "n_patients": len({e["patient_id"] for e in train})},
                "val":   {"n_images": len(val),   "n_patients": len({e["patient_id"] for e in val})},
                "test":  {"n_images": len(test),  "n_patients": len({e["patient_id"] for e in test})},
            },
            "removed_unreliable": list(UNRELIABLE_LABELS) if args.remove_unreliable else [],
            "seed": args.seed,
        }
        meta_path = os.path.join(args.out_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  -> {meta_path}")

    print(f"\n[DONE] All splits ready.")
    print(f"  Next: python scripts/03_validate.py --data-dir {args.out_dir}")


if __name__ == "__main__":
    main()