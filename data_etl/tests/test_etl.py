"""
NIH ChestX-ray14 ETL — Unit Tests
==================================
測試關鍵轉換邏輯，不需要真實資料。

覆蓋：
  1. GrayscaleTo3Channel transform
  2. get_transforms pipeline
  3. Label encoding / multi-hot
  4. CSV parsing (mock data)
  5. Patient-level split leakage check
  6. Class weight computation
  7. ChestXray14Dataset with mock images

用法：
  cd /path/to/workspace
  python -m pytest tests/test_etl.py -v
"""

import csv
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# 將 src 加入 path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chestxray14_dataset import (
    ALL_LABELS,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    N_CLASSES,
    GrayscaleTo3Channel,
    get_transforms,
    compute_class_weights,
    parse_metadata_csv,
    ChestXray14Dataset,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


# ══════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def mock_csv(tmp_path):
    """建立 mock CSV 檔案（清洗後格式）。"""
    csv_path = tmp_path / "mock.csv"
    rows = [
        {"image_index": "00000001_000.png", "patient_id": 1, "labels": "Atelectasis|Effusion", "age": 58, "gender": "M", "view": "PA"},
        {"image_index": "00000002_000.png", "patient_id": 1, "labels": "Atelectasis", "age": 58, "gender": "M", "view": "PA"},
        {"image_index": "00000003_000.png", "patient_id": 2, "labels": "", "age": 25, "gender": "F", "view": "AP"},
        {"image_index": "00000004_000.png", "patient_id": 2, "labels": "No Finding", "age": 25, "gender": "F", "view": "PA"},
        {"image_index": "00000005_000.png", "patient_id": 3, "labels": "Hernia|Pneumonia", "age": 72, "gender": "M", "view": "AP"},
        {"image_index": "00000006_000.png", "patient_id": 3, "labels": "Cardiomegaly|Edema|Effusion", "age": 72, "gender": "M", "view": "PA"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(csv_path)


@pytest.fixture
def mock_images(tmp_path):
    """建立 mock 影像檔案（8 張灰階 100x100 PNG）。"""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(1, 9):
        img = Image.fromarray(np.random.randint(0, 256, (100, 100), dtype=np.uint8), mode="L")
        img.save(img_dir / f"0000000{i}_000.png")
    return str(img_dir)


# ══════════════════════════════════════════════════════════════
# 1. Constants
# ══════════════════════════════════════════════════════════════

class TestConstants:
    def test_all_labels_count(self):
        assert len(ALL_LABELS) == 14

    def test_n_classes(self):
        assert N_CLASSES == 14

    def test_label_to_idx_completeness(self):
        assert len(LABEL_TO_IDX) == 14
        assert set(LABEL_TO_IDX.keys()) == set(ALL_LABELS)

    def test_idx_to_label_inversion(self):
        for lbl, idx in LABEL_TO_IDX.items():
            assert IDX_TO_LABEL[idx] == lbl

    def test_no_duplicates_in_labels(self):
        assert len(set(ALL_LABELS)) == len(ALL_LABELS)


# ══════════════════════════════════════════════════════════════
# 2. GrayscaleTo3Channel
# ══════════════════════════════════════════════════════════════

class TestGrayscaleTo3Channel:
    def test_output_shape(self):
        x = torch.randn(1, 224, 224)
        transform = GrayscaleTo3Channel()
        out = transform(x)
        assert out.shape == (3, 224, 224)

    def test_channel_values_identical(self):
        x = torch.randn(1, 64, 64)
        transform = GrayscaleTo3Channel()
        out = transform(x)
        torch.testing.assert_close(out[0], out[1])
        torch.testing.assert_close(out[1], out[2])

    def test_preserves_spatial(self):
        x = torch.randn(1, 128, 256)
        transform = GrayscaleTo3Channel()
        out = transform(x)
        assert out.shape[1] == 128
        assert out.shape[2] == 256


# ══════════════════════════════════════════════════════════════
# 3. get_transforms
# ══════════════════════════════════════════════════════════════

class TestGetTransforms:
    def test_train_transform_output_shape(self):
        transform = get_transforms("train", size=224)
        img = Image.fromarray(np.random.randint(0, 256, (300, 200), dtype=np.uint8), mode="L")
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_val_transform_output_shape(self):
        transform = get_transforms("val", size=224)
        img = Image.fromarray(np.random.randint(0, 256, (1024, 1024), dtype=np.uint8), mode="L")
        out = transform(img)
        assert out.shape == (3, 224, 224)

    def test_val_transform_deterministic(self):
        """Val/Test transforms 應為 deterministic（相同輸入 → 相同輸出）。"""
        transform = get_transforms("val", size=64)
        img = Image.fromarray(np.random.randint(0, 256, (100, 100), dtype=np.uint8), mode="L")
        out1 = transform(img)
        out2 = transform(img)
        torch.testing.assert_close(out1, out2)

    def test_custom_size(self):
        transform = get_transforms("train", size=128)
        img = Image.fromarray(np.random.randint(0, 256, (200, 200), dtype=np.uint8), mode="L")
        out = transform(img)
        assert out.shape == (3, 128, 128)

    def test_normalized_range(self):
        """Normalize 後各 channel 應為常數（均勻灰階輸入）。"""
        transform = get_transforms("val", size=64)
        # 均勻灰階 128 → ToTensor → 0.502 → repeat 3ch → normalize per-channel
        img = Image.fromarray(np.full((100, 100), 128, dtype=np.uint8), mode="L")
        out = transform(img)
        # Each channel should be constant (all pixels same value)
        for c in range(3):
            assert torch.allclose(out[c].mean(), out[c].min(), atol=1e-6)


# ══════════════════════════════════════════════════════════════
# 4. CSV Parsing
# ══════════════════════════════════════════════════════════════

class TestCSVParsing:
    def test_parse_cleaned_csv(self, mock_csv):
        entries = parse_metadata_csv(mock_csv)
        assert len(entries) == 6
        assert entries[0]["labels"] == ["Atelectasis", "Effusion"]
        assert entries[2]["labels"] == []  # empty string → No Finding
        assert entries[3]["labels"] == []  # "No Finding" → []
        assert entries[4]["patient_id"] == 3

    def test_parse_filters_invalid_labels(self, mock_csv):
        entries = parse_metadata_csv(mock_csv)
        for e in entries:
            for lbl in e["labels"]:
                assert lbl in LABEL_TO_IDX

    def test_parse_preserves_patient_ids(self, mock_csv):
        entries = parse_metadata_csv(mock_csv)
        pids = {e["patient_id"] for e in entries}
        assert pids == {1, 2, 3}

    def test_parse_kaggle_format(self, tmp_path):
        """測試原始 Kaggle CSV 格式。"""
        csv_path = tmp_path / "kaggle_style.csv"
        rows = [
            {"Image Index": "img1.png", "Finding Labels": "Atelectasis|Mass", "Patient ID": "10", "Patient Age": "45", "Patient Gender": "M", "View Position": "PA"},
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        entries = parse_metadata_csv(str(csv_path))
        assert len(entries) == 1
        assert entries[0]["labels"] == ["Atelectasis", "Mass"]
        assert entries[0]["patient_id"] == 10


# ══════════════════════════════════════════════════════════════
# 5. Multi-hot label encoding
# ══════════════════════════════════════════════════════════════

class TestMultiHotEncoding:
    def test_single_label(self, mock_csv, mock_images):
        ds = ChestXray14Dataset(csv_path=mock_csv, image_dir=mock_images, transform=get_transforms("val", size=64))
        # entry[0] has Atelectasis|Effusion → indices 0 and 2
        _, label = ds[0]
        assert label.shape == (14,)
        assert label[LABEL_TO_IDX["Atelectasis"]] == 1.0
        assert label[LABEL_TO_IDX["Effusion"]] == 1.0
        assert label[LABEL_TO_IDX["Mass"]] == 0.0

    def test_no_finding(self, mock_csv, mock_images):
        ds = ChestXray14Dataset(csv_path=mock_csv, image_dir=mock_images, transform=get_transforms("val", size=64))
        # entry[2] has no labels
        _, label = ds[2]
        assert label.sum() == 0.0

    def test_multi_label(self, mock_csv, mock_images):
        ds = ChestXray14Dataset(csv_path=mock_csv, image_dir=mock_images, transform=get_transforms("val", size=64))
        # entry[5] has Cardiomegaly|Edema|Effusion → 3 labels
        _, label = ds[5]
        assert label.sum() == 3.0
        assert label[LABEL_TO_IDX["Cardiomegaly"]] == 1.0
        assert label[LABEL_TO_IDX["Edema"]] == 1.0
        assert label[LABEL_TO_IDX["Effusion"]] == 1.0

    def test_label_dtype(self, mock_csv, mock_images):
        ds = ChestXray14Dataset(csv_path=mock_csv, image_dir=mock_images, transform=get_transforms("val", size=64))
        _, label = ds[0]
        assert label.dtype == torch.float32


# ══════════════════════════════════════════════════════════════
# 6. Patient-level split
# ══════════════════════════════════════════════════════════════

class TestPatientLevelSplit:
    def test_no_leakage(self):
        """用 mock entries 測試 patient-level split 無洩漏。"""
        entries = []
        for pid in range(100):
            for img_id in range(random.randint(1, 5)):
                labels = random.sample(ALL_LABELS, k=random.randint(0, 3))
                entries.append({
                    "image_index": f"{pid:08d}_{img_id:03d}.png",
                    "patient_id": pid,
                    "labels": labels,
                })

        random.seed(42)
        patient_ids = list({e["patient_id"] for e in entries})
        random.shuffle(patient_ids)

        n = len(patient_ids)
        n_train = int(n * 0.7)
        n_val = int(n * 0.1)

        train_pids = set(patient_ids[:n_train])
        val_pids = set(patient_ids[n_train:n_train + n_val])
        test_pids = set(patient_ids[n_train + n_val:])

        assert len(train_pids & val_pids) == 0
        assert len(train_pids & test_pids) == 0
        assert len(val_pids & test_pids) == 0

        # 確認所有患者都被分配
        all_assigned = train_pids | val_pids | test_pids
        assert all_assigned == set(patient_ids)


# ══════════════════════════════════════════════════════════════
# 7. Class weight computation
# ══════════════════════════════════════════════════════════════

class TestClassWeights:
    def test_median_freq_strategy(self):
        # 14 classes with varying frequency → ensure order is correct
        entries = []
        # Make all 14 classes appear with different frequencies
        for i, lbl in enumerate(ALL_LABELS):
            count = (len(ALL_LABELS) - i) * 10  # Atelectasis=140, Hernia=10
            entries.extend([{"labels": [lbl]} for _ in range(count)])
        weights = compute_class_weights(entries, strategy="median_freq")
        assert weights.shape == (14,)
        # Atelectasis (most common) should have lower weight than Hernia (least common)
        assert weights[LABEL_TO_IDX["Atelectasis"]] < weights[LABEL_TO_IDX["Hernia"]]
        assert weights[LABEL_TO_IDX["Hernia"]] > weights[LABEL_TO_IDX["Effusion"]]

    def test_all_positive(self):
        entries = [{"labels": [ALL_LABELS[0]]} for _ in range(50)]
        weights = compute_class_weights(entries, strategy="inverse")
        assert all(w > 0 for w in weights)

    def test_no_labels_still_works(self):
        entries = [{"labels": []} for _ in range(100)]
        weights = compute_class_weights(entries, strategy="inverse")
        assert weights.shape == (14,)
        # classes with 0 freq → very high weight (1/eps)
        assert all(w > 0 for w in weights)

    def test_balanced_strategy(self):
        entries = [
            {"labels": [lbl]} for lbl in ALL_LABELS for _ in range(10)
        ]
        weights = compute_class_weights(entries, strategy="balanced")
        # All classes equally frequent → all weights should be ~equal (after normalization)
        non_zero = weights[weights > 0]
        if len(non_zero) > 1:
            assert (non_zero.max() - non_zero.min()) < 0.01


# ══════════════════════════════════════════════════════════════
# 8. Dataset edge cases
# ══════════════════════════════════════════════════════════════

class TestDatasetEdgeCases:
    def test_missing_image_raises(self, mock_csv, tmp_path):
        ds = ChestXray14Dataset(csv_path=mock_csv, image_dir=str(tmp_path / "nonexistent"))
        with pytest.raises(FileNotFoundError):
            _ = ds[0]

    def test_return_meta(self, mock_csv, mock_images):
        ds = ChestXray14Dataset(
            csv_path=mock_csv, image_dir=mock_images,
            transform=get_transforms("val", size=64),
            return_meta=True,
        )
        img, label, meta = ds[0]
        assert "patient_id" in meta
        assert "age" in meta
        assert "gender" in meta
        assert "view" in meta
        assert "label_names" in meta
        assert isinstance(meta["label_names"], list)

    def test_dataset_length(self, mock_csv, mock_images):
        ds = ChestXray14Dataset(csv_path=mock_csv, image_dir=mock_images)
        assert len(ds) == 6

    def test_no_transform(self, mock_csv, mock_images):
        ds = ChestXray14Dataset(csv_path=mock_csv, image_dir=mock_images, transform=None)
        img, label = ds[0]
        # PIL Image without transform
        from PIL import Image as PILImage
        assert isinstance(img, PILImage.Image)
        assert img.mode == "L"  # still grayscale

    def test_entries_direct(self, mock_images):
        entries = [
            {"image_index": "00000001_000.png", "patient_id": 1, "labels": ["Atelectasis"], "age": 50, "gender": "M", "view": "PA"},
            {"image_index": "00000002_000.png", "patient_id": 1, "labels": [], "age": 50, "gender": "M", "view": "PA"},
        ]
        ds = ChestXray14Dataset(entries=entries, image_dir=mock_images)
        assert len(ds) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
