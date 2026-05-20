# NIH ChestX-ray14 ETL Pipeline

Complete data pipeline for NIH ChestX-ray14 dataset (Wang et al., CVPR 2017).

## Project Structure

```
t_831f6f29/
├── scripts/
│   ├── 01_download.sh          # 下載與解壓縮（Kaggle/TFDS/HuggingFace）
│   ├── 02_clean_metadata.py    # CSV 解析、清洗、patient-level split
│   ├── 03_validate_data.py     # 資料驗證（洩漏檢查、分佈統計、DataLoader 測試）
│   └── run_etl.sh              # 一鍵 ETL pipeline
├── src/
│   ├── __init__.py
│   └── chestxray14_dataset.py  # PyTorch Dataset / DataLoader / Transforms
├── tests/
│   └── test_etl.py             # 31 unit tests（mock data, no real dataset needed）
├── data/
│   ├── raw/                    # 原始下載資料
│   └── processed/              # 清洗後 CSV (train/val/test)
└── outputs/                    # 驗證報告等
```

## Quick Start

```bash
# 1. 下載 + 清洗 + 驗證 + 測試（一鍵）
bash scripts/run_etl.sh

# 2. 或分步執行
bash scripts/01_download.sh kaggle ./data/raw
python3 scripts/02_clean_metadata.py --csv data/raw/nih-chest-xrays/Data_Entry_2017_v2020.csv --out-dir data/processed
python3 scripts/03_validate_data.py --data-dir data/processed --img-dir data/raw/nih-chest-xrays/images

# 3. 只跑單元測試（不需要真實資料）
python3 -m pytest tests/test_etl.py -v
```

## Usage in Training Code

```python
from chestxray14_dataset import build_dataloaders, load_class_weights

# 一鍵建立 DataLoader
loaders = build_dataloaders(
    data_dir="data/processed",
    image_dir="data/raw/nih-chest-xrays/images",
    batch_size=32,
    size=224,
    use_clahe=False,
)

train_loader = loaders["train"]   # shuffle=True, augmentation
val_loader = loaders["val"]       # no augmentation
test_loader = loaders["test"]     # no augmentation

# Class weights for BCEWithLogitsLoss
weights = load_class_weights("data/processed/class_weights.json")
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
```

## Key Design Decisions

1. **Patient-level split** — 嚴格防止 train/val/test 洩漏（官方 split 有此問題）
2. **Multi-hot encoding** — 14 類，支援每張影像多個標籤
3. **Grayscale → 3 channels** — repeat 3x for ImageNet pretrained backbone
4. **Median-freq class weights** — 預設策略，處理嚴重不平衡（Hernia 0.2% vs Effusion 12.2%）
5. **Data augmentation** — train: horizontal flip + rotation(±10°) + affine + color jitter
6. **Infiltration label** — 標記為不可靠但預設不移除（可選 --remove-unreliable）

## Dependencies

- Python 3.8+
- PyTorch >= 1.9
- torchvision
- Pillow
- numpy
- opencv-python (optional, for CLAHE)
- pytest (for tests)
