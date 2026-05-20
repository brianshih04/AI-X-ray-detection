"""
Chest X-ray Dataset Module
支援 CheXpert 多標籤分類，含 patient-level split 避免資料洩漏。
"""

import os
import logging
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

CHEXPERT_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
    "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices",
]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(cfg_aug, img_size: int, split: str = "train"):
    """Build train/val transforms from config augmentation section."""
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        ops = [transforms.Resize((img_size, img_size))]

        if cfg_aug.get("random_horizontal_flip", 0.5) > 0:
            ops.append(transforms.RandomHorizontalFlip(p=cfg_aug["random_horizontal_flip"]))

        if cfg_aug.get("random_affine", False):
            ops.append(transforms.RandomAffine(
                degrees=cfg_aug.get("random_rotation", 10),
                translate=cfg_aug.get("random_translate", [0.05, 0.05]),
                scale=cfg_aug.get("random_scale", [0.9, 1.1]),
            ))

        if cfg_aug.get("color_jitter", 0) > 0:
            ops.append(transforms.ColorJitter(
                brightness=cfg_aug["color_jitter"],
                contrast=cfg_aug["color_jitter"],
            ))

        ops.extend([
            transforms.ToTensor(),
            normalize,
        ])
        if cfg_aug.get("random_erasing", 0) > 0:
            ops.append(transforms.RandomErasing(p=cfg_aug["random_erasing"]))

        return transforms.Compose(ops)

    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


class ChestXrayDataset(Dataset):
    """CheXpert multi-label dataset.

    Handles:
    - Uncertain labels (U = -1): zero / one / random
    - Patient-level ID extraction from Path
    - Graceful handling of missing/corrupt images
    """

    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        label_columns: List[str],
        transform=None,
        uncertain_policy: str = "zero",
    ):
        self.data_dir = data_dir
        self.transform = transform
        self.label_columns = label_columns

        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"Raw CSV: {len(df)} rows from {csv_path}")

        # Extract patient ID from CheXpert Path
        # Path format: CheXpert-v1.0-small/patient00001/study1/view1_frontal.jpg
        if "Path" in df.columns:
            df["PatientID"] = df["Path"].apply(
                lambda p: p.split("/")[1] if "/" in str(p) else "unknown"
            )

        # Filter rows with valid paths
        df = df[df["Path"].notna()].reset_index(drop=True)

        # Process label columns — fill NaN with 0, handle uncertain (-1)
        for col in self.label_columns:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = df[col].fillna(0.0)
            if uncertain_policy == "zero":
                df.loc[df[col] == -1.0, col] = 0.0
            elif uncertain_policy == "one":
                df.loc[df[col] == -1.0, col] = 1.0
            elif uncertain_policy == "random":
                mask = df[col] == -1.0
                df.loc[mask, col] = np.random.choice(
                    [0.0, 1.0], size=mask.sum()
                ).astype(float)

        self.df = df
        logger.info(
            f"Dataset ready: {len(self.df)} samples, "
            f"{len(self.label_columns)} labels, "
            f"uncertain_policy={uncertain_policy}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row["Path"])

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}, using black image")
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        labels = torch.tensor(
            row[self.label_columns].values.astype(np.float32),
            dtype=torch.float32,
        )
        return img, labels


def patient_level_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe by patient ID to prevent data leakage."""
    patient_ids = df["PatientID"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(patient_ids)

    n_test = max(1, int(len(patient_ids) * test_size))
    test_patients = set(patient_ids[:n_test])

    train_df = df[~df["PatientID"].isin(test_patients)].reset_index(drop=True)
    val_df = df[df["PatientID"].isin(test_patients)].reset_index(drop=True)
    logger.info(
        f"Patient-level split: train={len(train_df)} (pts={df['PatientID'].nunique() - n_test}), "
        f"val={len(val_df)} (pts={n_test})"
    )
    return train_df, val_df


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, List[str]]:
    """Build train/val dataloaders from config. Returns (train_loader, val_loader, label_columns)."""
    data_cfg = cfg["data"]
    label_columns = cfg.get("labels", CHEXPERT_LABELS)
    img_size = data_cfg["img_size"]

    train_transform = get_transforms(cfg.get("augmentation", {}), img_size, "train")
    val_transform = get_transforms(cfg.get("augmentation", {}), img_size, "val")

    train_csv = os.path.join(data_cfg["data_dir"], data_cfg["train_csv"])
    val_csv = os.path.join(data_cfg["data_dir"], data_cfg["valid_csv"])

    train_ds = ChestXrayDataset(
        csv_path=train_csv,
        data_dir=data_cfg["data_dir"],
        label_columns=label_columns,
        transform=train_transform,
        uncertain_policy=data_cfg.get("uncertain_policy", "zero"),
    )

    val_ds = ChestXrayDataset(
        csv_path=val_csv,
        data_dir=data_cfg["data_dir"],
        label_columns=label_columns,
        transform=val_transform,
        uncertain_policy=data_cfg.get("uncertain_policy", "zero"),
    )

    common = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    return train_loader, val_loader, label_columns
