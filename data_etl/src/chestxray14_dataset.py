"""
NIH ChestX-ray14 — PyTorch Dataset / DataLoader

影像前處理 pipeline + PyTorch Dataset 實作，支援：
  - 灰階 PNG 讀取 → 3-channel repeat → ImageNet normalization
  - Resize 224x224
  - Data augmentation（訓練時：水平翻轉、旋轉、平移、亮度/對比抖動）
  - CLAHE 對比度增強（可選）
  - Patient-level train/val/test split
  - Multi-hot label encoding（14 類）
  - Weighted BCE 權重計算

參考：
  Wang et al. CVPR 2017, Rajpurkar et al. (CheXNet) 2017
  T1 分析報告：NIH_ChestXray14_Analysis.md
"""

import copy
import json
import os
import random
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ══════════════════════════════════════════════════════════════
# 1. 常數定義
# ══════════════════════════════════════════════════════════════

ALL_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Consolidation",
    "Edema", "Hernia",
]

LABEL_TO_IDX = {label: i for i, label in enumerate(ALL_LABELS)}
IDX_TO_LABEL = {i: label for label, i in LABEL_TO_IDX.items()}
N_CLASSES = len(ALL_LABELS)  # 14

# ImageNet normalization（用於 pretrained backbone）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# T1 分析建議：Infiltration 標籤不可靠（多篇論文建議移除）
# 預設不移除，由使用者 --remove-unreliable 啟動
UNRELIABLE_LABELS = {"Infiltration"}


# ══════════════════════════════════════════════════════════════
# 2. 影像前處理 Transforms
# ══════════════════════════════════════════════════════════════

class GrayscaleTo3Channel:
    """灰階 1ch → 3ch repeat（在 ToTensor 之後用）。"""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (1, H, W) → (3, H, W)
        return x.repeat(3, 1, 1)


class CLAHETransform:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization)
    在 PIL Image 上操作（ToTensor 之前）。

    Args:
        clip_limit: 對比度限制，預設 2.0
        tile_grid_size: 格柵大小，預設 (8, 8)
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        import cv2
        arr = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        arr = clahe.apply(arr)
        return Image.fromarray(arr)


def get_transforms(
    split: str = "train",
    size: int = 224,
    use_clahe: bool = False,
    clahe_clip_limit: float = 2.0,
    brightness_jitter: float = 0.2,
    contrast_jitter: float = 0.2,
) -> transforms.Compose:
    """
    取得 transforms pipeline。

    Args:
        split: "train" / "val" / "test"
        size: 輸出尺寸（預設 224x224）
        use_clahe: 是否啟用 CLAHE 對比度增強
        clahe_clip_limit: CLAHE clip limit
        brightness_jitter: 亮度抖動幅度
        contrast_jitter: 對比度抖動幅度

    Returns:
        torchvision.transforms.Compose
    """
    steps = []

    # CLAHE（可選，在 resize 之前）
    if use_clahe:
        steps.append(CLAHETransform(clip_limit=clahe_clip_limit))

    # Resize
    steps.append(transforms.Resize((size, size)))

    if split == "train":
        # Data augmentation
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
            ),
            transforms.ColorJitter(
                brightness=brightness_jitter,
                contrast=contrast_jitter,
            ),
        ])

    # ToTensor（PIL → Tensor, [0,255] → [0.0,1.0]）
    steps.append(transforms.ToTensor())

    # Grayscale 1ch → 3ch（for ImageNet backbone）
    steps.append(GrayscaleTo3Channel())

    # ImageNet normalization
    steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(steps)


# ══════════════════════════════════════════════════════════════
# 3. CSV 解析
# ══════════════════════════════════════════════════════════════

def parse_metadata_csv(csv_path: str) -> List[Dict]:
    """
    解析清洗後的 metadata CSV（由 02_clean_metadata.py 產出）。
    支援兩種格式：
      - 原始格式（Kaggle CSV）
      - 清洗後格式（02_clean_metadata.py 輸出）
    """
    import csv

    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        for row in reader:
            # 清洗後的格式有 labels 欄位（| 分隔）
            if "labels" in fieldnames:
                labels_str = row.get("labels", "").strip()
                if labels_str == "" or labels_str == "No Finding":
                    labels = []
                else:
                    labels = [l.strip() for l in labels_str.split("|") if l.strip()]
            else:
                # 原始 Kaggle CSV 格式
                labels_str = row.get("Finding Labels", "No Finding").strip()
                if labels_str == "No Finding" or labels_str == "":
                    labels = []
                else:
                    labels = [l.strip() for l in labels_str.split("|")]

            try:
                patient_id = int(row.get("patient_id", row.get("Patient ID", "0")))
            except (ValueError, TypeError):
                patient_id = 0

            image_index = row.get("image_index", row.get("Image Index", "")).strip()

            if not image_index:
                continue

            # 過濾不合法的標籤
            labels = [l for l in labels if l in LABEL_TO_IDX]

            try:
                age = int(row.get("age", row.get("Patient Age", "0")))
                if age < 0 or age > 120:
                    age = 0
            except (ValueError, TypeError):
                age = 0

            gender = row.get("gender", row.get("Patient Gender", "Unknown")).strip()
            if gender not in ("M", "F"):
                gender = "Unknown"

            view = row.get("view", row.get("View Position", "Unknown")).strip()
            if view not in ("AP", "PA"):
                view = "Unknown"

            entries.append({
                "image_index": image_index,
                "patient_id": patient_id,
                "labels": labels,
                "age": age,
                "gender": gender,
                "view": view,
            })

    return entries


# ══════════════════════════════════════════════════════════════
# 4. Class Weight 計算
# ══════════════════════════════════════════════════════════════

def compute_class_weights(
    entries: List[Dict],
    strategy: str = "median_freq",
) -> torch.Tensor:
    """
    計算 14 類的 per-class 權重（用於 BCEWithLogitsLoss pos_weight）。

    Strategies:
      "inverse":     w_i = 1 / freq_i
      "median_freq": w_i = median(freq) / freq_i    (T1 推薦)
      "sqrt":        w_i = 1 / sqrt(freq_i)
      "effective":   w_i = (1 - freq_i) / freq_i     (effective number of samples)

    Returns:
        torch.Tensor of shape (14,)
    """
    counter = Counter()
    for e in entries:
        for lbl in e["labels"]:
            if lbl in LABEL_TO_IDX:
                counter[LABEL_TO_IDX[lbl]] += 1

    total = len(entries)
    freqs = np.zeros(N_CLASSES)
    for i in range(N_CLASSES):
        freqs[i] = counter.get(i, 0) / max(total, 1)

    if strategy == "inverse":
        weights = 1.0 / np.maximum(freqs, 1e-6)
    elif strategy == "median_freq":
        median = np.median(freqs)
        weights = median / np.maximum(freqs, 1e-6)
    elif strategy == "sqrt":
        weights = 1.0 / np.sqrt(np.maximum(freqs, 1e-6))
    elif strategy == "effective":
        weights = (1.0 - freqs) / np.maximum(freqs, 1e-6)
    elif strategy == "balanced":
        weights = (total * (1.0 - freqs)) / (N_CLASSES * np.maximum(freqs, 1e-6))
        weights = weights / weights.max()  # normalize to [0, 1]
    else:
        raise ValueError(f"Unknown weight strategy: {strategy}")

    return torch.tensor(weights, dtype=torch.float32)


def load_class_weights(json_path: str) -> torch.Tensor:
    """從 JSON 檔載入 class weights。"""
    with open(json_path) as f:
        d = json.load(f)
    weights = np.zeros(N_CLASSES)
    for lbl, w in d.items():
        if lbl in LABEL_TO_IDX:
            weights[LABEL_TO_IDX[lbl]] = w
    return torch.tensor(weights, dtype=torch.float32)


# ══════════════════════════════════════════════════════════════
# 5. PyTorch Dataset
# ══════════════════════════════════════════════════════════════

class ChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14 PyTorch Dataset

    支援從清洗後的 CSV 或直接傳入 entries list 建立。

    Args:
        csv_path:     metadata CSV 路徑（train.csv / val.csv / test.csv）
        image_dir:    影像目錄路徑
        entries:      已分割的 entries list（若提供則忽略 csv_path）
        transform:    torchvision transforms pipeline
        return_meta:  是否回傳 metadata dict

    Returns:
        image: torch.Tensor (3, H, W) — 已 normalize
        label: torch.Tensor (14,) — multi-hot
        meta:  dict（僅 return_meta=True）
    """

    def __init__(
        self,
        csv_path: Optional[str] = None,
        image_dir: str = "data/raw/nih-chest-xrays/images",
        entries: Optional[List[Dict]] = None,
        transform: Optional[Callable] = None,
        return_meta: bool = False,
    ):
        if entries is not None:
            self.entries = entries
        elif csv_path is not None:
            self.entries = parse_metadata_csv(csv_path)
        else:
            raise ValueError("Provide either csv_path or entries")

        self.image_dir = Path(image_dir)
        self.transform = transform
        self.return_meta = return_meta

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        img_path = self.image_dir / entry["image_index"]

        # 讀取灰階 PNG
        try:
            image = Image.open(img_path).convert("L")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Image not found: {img_path}\n"
                f"  image_index={entry['image_index']}, "
                f"patient_id={entry['patient_id']}"
            )

        if self.transform is not None:
            image = self.transform(image)

        # Multi-hot label encoding (14 classes)
        label = torch.zeros(N_CLASSES, dtype=torch.float32)
        for lbl in entry["labels"]:
            label[LABEL_TO_IDX[lbl]] = 1.0

        if self.return_meta:
            meta = {
                "patient_id": entry["patient_id"],
                "age": entry["age"],
                "gender": entry["gender"],
                "view": entry["view"],
                "label_names": entry["labels"],
                "image_index": entry["image_index"],
            }
            return image, label, meta
        return image, label


# ══════════════════════════════════════════════════════════════
# 6. DataLoader Factory
# ══════════════════════════════════════════════════════════════

def build_dataloaders(
    data_dir: str,
    image_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    size: int = 224,
    use_clahe: bool = False,
    pin_memory: bool = True,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    一鍵建立 train / val / test DataLoader。

    Args:
        data_dir:    清洗後 CSV 所在目錄（含 train.csv, val.csv, test.csv）
        image_dir:   影像目錄路徑
        batch_size:  batch 大小
        num_workers: DataLoader workers
        size:        影像尺寸
        use_clahe:   是否啟用 CLAHE
        pin_memory:  是否 pin memory（GPU 訓練建議 True）
        seed:        random seed

    Returns:
        {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    # Transforms
    train_transform = get_transforms("train", size=size, use_clahe=use_clahe)
    val_transform = get_transforms("val", size=size, use_clahe=use_clahe)

    # Datasets
    train_ds = ChestXray14Dataset(
        csv_path=os.path.join(data_dir, "train.csv"),
        image_dir=image_dir,
        transform=train_transform,
    )
    val_ds = ChestXray14Dataset(
        csv_path=os.path.join(data_dir, "val.csv"),
        image_dir=image_dir,
        transform=val_transform,
    )
    test_ds = ChestXray14Dataset(
        csv_path=os.path.join(data_dir, "test.csv"),
        image_dir=image_dir,
        transform=val_transform,  # test 同 val transform
    )

    # DataLoaders
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def seed_worker(worker_id: int) -> None:
    """DataLoader worker seed，確保可重現性。"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ══════════════════════════════════════════════════════════════
# 7. CLI / 快速驗證
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChestXray14 Dataset quick test")
    parser.add_argument("--csv", type=str, help="Path to a metadata CSV")
    parser.add_argument("--img-dir", type=str, default="data/raw/nih-chest-xrays/images")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--clahe", action="store_true")
    args = parser.parse_args()

    if args.csv:
        # 從單一 CSV 測試
        transform = get_transforms("train", size=args.size, use_clahe=args.clahe)
        ds = ChestXray14Dataset(csv_path=args.csv, image_dir=args.img_dir, transform=transform)
        print(f"Dataset: {len(ds)} images")

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        img, label = next(iter(loader))
        print(f"Batch image shape: {img.shape}")   # (B, 3, 224, 224)
        print(f"Batch label shape: {label.shape}")  # (B, 14)
        print(f"Label example: {label[0].tolist()}")
        print(f"Label names: {[IDX_TO_LABEL[i] for i, v in enumerate(label[0].tolist()) if v == 1.0]}")
    else:
        print("Usage: python chestxray14_dataset.py --csv <path> [--img-dir <dir>]")
        print("  Or use build_dataloaders() with the full ETL pipeline output.")
