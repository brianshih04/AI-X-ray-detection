"""
NIH ChestX-ray14 資料準備指南 — 可直接執行的 PyTorch Dataset

使用前請先下載資料集：
    kaggle datasets download -d nih-chest-xrays -p ./data/
    unzip data/nih-chest-xrays.zip -d data/nih-chest-xrays/

本 script 提供完整的 Dataset class + patient-level split + 分析工具。
"""

import os
import csv
import random
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================
# 1. 常數定義
# ============================================================

ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Consolidation",
    "Edema", "Hernia",
]

LABEL_TO_IDX = {label: i for i, label in enumerate(ALL_LABELS)}

# Weighted BCE 權重（需根據實際資料計算，此為範例）
# 建議在 fit() 後使用下方 compute_class_weights() 取得
DEFAULT_CLASS_WEIGHTS = None  # 設為 None 會在 init 時自動計算


# ============================================================
# 2. CSV 解析
# ============================================================

def parse_csv(csv_path: str) -> List[Dict]:
    """解析 Data_Entry_2017_v2020.csv"""
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels_str = row["Finding Labels"]
            if labels_str == "No Finding":
                labels = []
            else:
                labels = labels_str.split("|")
            entries.append({
                "image_index": row["Image Index"],
                "labels": labels,
                "patient_id": int(row["Patient ID"]),
                "age": int(row["Patient Age"]),
                "gender": row["Patient Gender"],
                "view": row["View Position"],
            })
    return entries


def patient_level_split(
    entries: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.10,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Patient-level split — 避免 train/test 資料洩漏。
    同一病患的所有影像只會出現在同一個 split。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # 按 patient_id 分組
    patient_to_indices: Dict[int, List[int]] = {}
    for i, entry in enumerate(entries):
        pid = entry["patient_id"]
        patient_to_indices.setdefault(pid, []).append(i)

    patient_ids = list(patient_to_indices.keys())
    random.seed(seed)
    random.shuffle(patient_ids)

    n_total = len(patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train : n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val :])

    train_entries = [e for e in entries if e["patient_id"] in train_patients]
    val_entries = [e for e in entries if e["patient_id"] in val_patients]
    test_entries = [e for e in entries if e["patient_id"] in test_patients]

    return train_entries, val_entries, test_entries


def compute_class_weights(
    entries: List[Dict],
    strategy: str = "median_freq",
) -> torch.Tensor:
    """
    計算 per-class 權重。
    
    Strategies:
        "inverse":     w_i = 1 / freq_i
        "median_freq": w_i = median_freq / freq_i
        "sqrt":        w_i = 1 / sqrt(freq_i)
    """
    counter = Counter()
    for entry in entries:
        for label in entry["labels"]:
            counter[LABEL_TO_IDX[label]] += 1

    total = sum(counter.values())
    n_classes = len(ALL_LABELS)

    freqs = torch.zeros(n_classes)
    for i in range(n_classes):
        freqs[i] = counter.get(i, 0) / max(total, 1)

    if strategy == "inverse":
        weights = 1.0 / freqs.clamp(min=1e-6)
    elif strategy == "median_freq":
        median = freqs.median()
        weights = median / freqs.clamp(min=1e-6)
    elif strategy == "sqrt":
        weights = 1.0 / freqs.clamp(min=1e-6).sqrt()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return weights


# ============================================================
# 3. PyTorch Dataset
# ============================================================

class ChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14 PyTorch Dataset
    
    Args:
        csv_path:     Data_Entry_2017_v2020.csv 路徑
        image_dir:    images/ 資料夾路徑
        entries:      已分割的 entries list（若提供則忽略 csv_path）
        transform:    torchvision transforms
        multi_label:  True=multi-hot (14), False=single label
        return_meta:  是否回傳 metadata dict
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        image_dir: str = "data/nih-chest-xrays/images",
        entries: Optional[List[Dict]] = None,
        transform=None,
        multi_label: bool = True,
        return_meta: bool = False,
    ):
        if entries is not None:
            self.entries = entries
        elif csv_path is not None:
            self.entries = parse_csv(csv_path)
        else:
            raise ValueError("Provide either csv_path or entries")

        self.image_dir = Path(image_dir)
        self.transform = transform
        self.multi_label = multi_label
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        img_path = self.image_dir / entry["image_index"]

        # Load grayscale PNG
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        if self.multi_label:
            label = torch.zeros(len(ALL_LABELS), dtype=torch.float32)
            for l in entry["labels"]:
                label[LABEL_TO_IDX[l]] = 1.0
        else:
            if len(entry["labels"]) == 0:
                label = torch.tensor(-1, dtype=torch.long)  # "No Finding"
            else:
                label = torch.tensor(LABEL_TO_IDX[entry["labels"][0]], dtype=torch.long)

        if self.return_meta:
            meta = {
                "patient_id": entry["patient_id"],
                "age": entry["age"],
                "gender": entry["gender"],
                "view": entry["view"],
                "labels": entry["labels"],
            }
            return image, label, meta
        return image, label


# ============================================================
# 4. Transforms
# ============================================================

def get_transforms(split: str = "train", size: int = 224):
    """
    取得 transforms pipeline。
    
    - 灰階 → repeat 3 channels → ImageNet normalize
    - 訓練時加入 data augmentation
    """
    if split == "train":
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 1ch → 3ch
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transform


# ============================================================
# 5. 使用範例
# ============================================================

if __name__ == "__main__":
    # --- 快速分析 ---
    CSV_PATH = "data/nih-chest-xrays/Data_Entry_2017_v2020.csv"
    IMG_DIR = "data/nih-chest-xrays/images"

    all_entries = parse_csv(CSV_PATH)
    print(f"Total images: {len(all_entries)}")
    print(f"Unique patients: {len(set(e['patient_id'] for e in all_entries))}")

    # 標籤分佈
    label_counter = Counter()
    for e in all_entries:
        for l in e["labels"]:
            label_counter[l] += 1
    print("\nLabel distribution:")
    for label, count in label_counter.most_common():
        pct = count / len(all_entries) * 100
        print(f"  {label:25s}: {count:>7d} ({pct:5.1f}%)")

    # Patient-level split
    train, val, test = patient_level_split(all_entries)
    print(f"\nPatient-level split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Class weights
    weights = compute_class_weights(train, strategy="median_freq")
    print("\nMedian-freq class weights:")
    for label, w in zip(ALL_LABELS, weights.tolist()):
        print(f"  {label:25s}: {w:.3f}")

    # --- DataLoader 範例 ---
    train_ds = ChestXray14Dataset(
        entries=train,
        image_dir=IMG_DIR,
        transform=get_transforms("train", size=224),
        multi_label=True,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

    # 確認 shape
    img, label = next(iter(train_loader))
    print(f"\nBatch shape: {img.shape}")    # [32, 3, 224, 224]
    print(f"Label shape: {label.shape}")   # [32, 14]
    print(f"Label example: {label[0].tolist()}")
