# 胸腔 X 光影像分類：從零到上線的完整訓練指南

> 目標受眾：沒有 ML 經驗的開發者 or 醫療 domain expert  
> 目標：跟著本指南跑完第一個胸腔 X 光多標籤分類模型  
> 最後更新：2026-05

---

## 目錄

1. [前置準備：了解你要解決的問題](#1-前置準備了解你要解決的問題)
2. [模型架構推薦](#2-模型架構推薦)
3. [Transfer Learning 策略](#3-transfer-learning-策略)
4. [訓練配置建議](#4-訓練配置建議)
5. [多標籤分類的特殊考量](#5-多標籤分類的特殊考量)
6. [PyTorch 完整訓練程式碼框架](#6-pytorch-完整訓練程式碼框架)
7. [GPU 需求與雲端訓練方案](#7-gpu-需求與雲端訓練方案)
8. [常見陷阱與除錯指南](#8-常見陷阱與除錯指南)
9. [快速上路懶人包](#9-快速上路懶人包)

---

## 1. 前置準備：了解你要解決的問題

### 胸腔 X 光分類是什麼？

輸入：一張 Chest X-ray（胸腔正面或側面 X 光片）  
輸出：多個標籤同時為 True/False（例如：肺結節、肺炎、氣胸、心臟肥大……）

### 資料集選擇

| 資料集 | 影像數 | 標籤數 | 特色 | 授權 |
|--------|--------|--------|------|------|
| **CheXpert** (Stanford, 2019) | 224,316 | 14 | 最大、最常用 | 研究用免費 |
| **MIMIC-CXR** (MIT, 2019) | 377,110 | 14 | 含臨床文字報告 | 需申請 |
| **NIH ChestX-ray14** (NIH, 2017) | 112,120 | 14 | 最早、爭議多（label noise） | 開放 |
| **VinDr-CXR** (Vietnam, 2020) | 20,000 | 28 | 品質佳、有 bounding box | 研究用 |

**推薦：從 CheXpert 開始**，影像品質穩定、label 爭議較少，文獻最多。

### CheXpert 14 個標籤

```
Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum,
Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion,
Pleural Other, Pneumonia, Pneumothorax, Support Devices
```

### 硬體需求初估

| 資料集大小 | 最低 GPU | 建議 GPU | 訓練時間（估） |
|-----------|----------|----------|---------------|
| 10k 張（subset） | RTX 3060 12GB | RTX 4090 24GB | 1-3 小時 |
| 100k 張（full CheXpert） | RTX 4090 24GB | A100 40GB | 8-24 小時 |
| 300k+ 張（MIMIC） | A100 40GB | A100 80GB | 1-3 天 |

---

## 2. 模型架構推薦

### 2.1 三種主流架構比較

```
┌─────────────────────────────────────────────────────────────────────┐
│                    模型架構比較：胸腔 X 光分類                         │
├───────────────┬──────────────┬──────────────┬──────────────────────┤
│     架構       │  參數數量     │  推論速度     │  優缺點               │
├───────────────┼──────────────┼──────────────┼──────────────────────┤
│ DenseNet-121  │  8M          │  ★★★★        │ 經典、醫療文獻最多、   │
│ (baseline)    │              │  (中等)       │ 特徵重用好、訓練穩定   │
├───────────────┼──────────────┼──────────────┼──────────────────────┤
│ EfficientNet-B3│  12M        │  ★★★★★       │ 準確率更高、參數效率好、│
│               │              │  (快)         │ PyTorch 支援好        │
├───────────────┼──────────────┼──────────────┼──────────────────────┤
│ ViT-B/16      │  86M         │  ★★          │ 對大資料集效果好、      │
│               │              │  (慢)         │ 小資料集容易過擬合      │
│               │              │              │ 需要較多資料增強        │
└───────────────┴──────────────┴──────────────┴──────────────────────┘
```

### 2.2 推薦策略

**新手推薦順序（由易到難）：**

1. **EfficientNet-B0**（最推薦起點）
   - 參數少、訓練快、準確率佳
   - `torchvision.models.efficientnet_b0(weights='IMAGENET1K_V1')`
   - 在 CheXpert 上 AUC 可達 0.85+

2. **DenseNet-121**（學術標竿）
   - 論文引用最多，實作參考豐富
   - `torchvision.models.densenet121(weights='IMAGENET1K_V1')`
   - 適合需要對比文獻結果的場景

3. **EfficientNet-B3**（進階提升）
   - B0 跑順之後換 B3 進一步提升
   - 需更多 GPU 記憶體（建議 16GB+）

4. **ViT-B/16**（研究探索）
   - 需要更多技巧（更 aggressive 的 augmentation）
   - 建議已有經驗後再試

### 2.3 Medical Foundation Models（推薦！）

不要再從頭 train backbone 了，用醫療領域預訓練模型：

| 模型 | 來源 | 特色 |
|------|------|------|
| **MedCLIP** | Stanford | CLIP-based, 跨模態（影像+文字） |
| **BioViL** | Microsoft | Chest X-ray 專用 transformers |
| **CheXzero** | NYU | Zero-shot chest X-ray classification |
| **RadImageNet** | Stanford | 專為放射科影像設計的預訓練權重 |

**使用方式（以 BioViL 為例）：**
```python
# pip install biomedical-vl
from biomedical_vl import BioViLVLModel

model = BioViLVLModel.from_pretrained("microsoft/biomed-vil")
# 替換分類頭層
model.classifier = nn.Linear(model.config.hidden_size, 14)
```

---

## 3. Transfer Learning 策略

### 3.1 四種 Transfer Learning 等級

```
從易到難：

Level 1: 全部凍結，只 train 分類頭（最簡單，適合新手）
─────────────────────────────────────────────────────────
模型 backbone (ImageNet weights) ──[freeze]──> 特徵提取
                                             ↓
                                       新增分類頭 (隨機初始化)
                                             ↓
                                    只更新分類頭的 weights
─────────────────────────────────────────────────────────

Level 2: 部分解凍（常見做法）
─────────────────────────────────────────────────────────
backbone 前半 [freeze] ──> 深層 [unfreeze with small LR]
                                             ↓
                                    全部一起 train（backbone LR 很小）
─────────────────────────────────────────────────────────

Level 3: Progressive Unfreezing（漸進式解凍）
─────────────────────────────────────────────────────────
Step 1: train 20 epochs 只更新分類頭
Step 2: unfreeze 後 2 層，train 20 epochs（backbone LR = head LR × 0.1）
Step 3: unfreeze 全 backbone，train 到收斂（backbone LR = head LR × 0.01）
─────────────────────────────────────────────────────────

Level 4: Full Fine-tuning（你有 GPU + 資料充足）
─────────────────────────────────────────────────────────
backbone + head 全部 train
通常需要 warmup + 小的 learning rate
```

### 3.2 推薦：分層 Learning Rate

```python
import torch.optim as optim

# backbone 用小 LR，頭層用大 LR
optimizer = optim.AdamW([
    {'params': model.features.parameters(), 'lr': 1e-5},    # backbone: 極小
    {'params': model.classifier.parameters(), 'lr': 1e-3},  # 頭層: 正常
], weight_decay=0.01)
```

### 3.3 Domain Adaptation：利用 CheXpert 預訓練權重

```python
# 不要用 ImageNet 預訓練，直接用 CheXpert 預訓練權重
# 模型下載：https://github.com/MIC-DKFZ/CheXNet
# 或使用 MedCLIP 的 CheXpert-fine-tuned weights

import torch
model = torchvision.models.densenet121(weights=None)  # 不要 IMAGENET1K

# 下載 CheXNet 權重
state_dict = torch.load('chexnet_densenet121.pth', map_location='cpu')
model.load_state_dict(state_dict, strict=False)

# 替換分類層為 14 輸出的 multi-label head
model.classifier = nn.Linear(model.classifier.in_features, 14)
```

---

## 4. 訓練配置建議

### 4.1 Loss Function

```python
import torch
import torch.nn as nn

# 標準選擇：Binary Cross-Entropy（multi-label 必用）
criterion = nn.BCEWithLogitsLoss()

# 類別不平衡嚴重時用 Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

### 4.2 Optimizer：AdamW（幾乎總是最優選擇）

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,           # head learning rate
    weight_decay=0.01, # L2 正則化（比 Adam 的 L2 更穩定）
    betas=(0.9, 0.999)
)
```

### 4.3 Learning Rate Schedule

```python
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # 第一個 cycle 長度（epoch）
    T_mult=2,      # 之後每個 cycle × 2（越來越長）
    eta_min=1e-6   # 最低 LR
)

# 或者更簡單：StepLR（新手推薦）
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)
```

### 4.4 Batch Size 參考

| GPU VRAM | Batch Size | 備註 |
|----------|-----------|------|
| 6GB  | 8-16 | DenseNet-121, 可能需要 gradient accumulation |
| 12GB | 16-32 | 標準實驗設定 |
| 24GB | 32-64 | RTX 4090 推薦 |
| 40GB | 64-128 | A100 推薦 |
| 80GB | 128-256 | 完整 MIMIC 訓練 |

```python
# 記憶體不夠？用 Gradient Accumulation
accum_steps = 4
effective_batch = batch_size * accum_steps
# 4 步才更新一次，達到 4 倍 batch size 的效果
```

### 4.5 Data Augmentation（胸腔 X 光專用）

```python
import torchvision.transforms as T

train_transform = T.Compose([
    T.Resize((224, 224)),           # ViT 需要 224×224
    T.RandomHorizontalFlip(p=0.5),   # 水平翻轉（X 光通常對稱）
    T.RandomAffine(
        degrees=15,                  # 小角度旋轉
        translate=(0.1, 0.1),        # 平移
        scale=(0.9, 1.1)             # 縮放
    ),
    T.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**注意：不要用太多 augmentation**，胸腔 X 光差異細微，強 augmentation 會破壞病理特徵。

### 4.6 MixUp / CutMix

對醫療影像要保守使用：

```python
# 簡化版 MixUp（避免破壞病理性質）
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y
```

---

## 5. 多標籤分類的特殊考量

### 5.1 Sigmoid vs Softmax

```
Softmax：輸出加起來 = 1，適用於「只選一個類別」（互斥）
         例如：貓、狗、鳥 → 只能選一個

Sigmoid：每個類別獨立伯努利，適用於「可以同時是多個」（不互斥）
         例如：胸腔 X 光 → 可以同時有「肺炎」+「肺積水」+「心臟肥大」
```

```python
# 正確做法：用 Sigmoid（multi-label）
outputs = model(images)  # shape: [batch, 14]
probs = torch.sigmoid(outputs)  # 每個類別獨立的 0~1 機率

# 錯誤做法：用 Softmax（會強制類別互斥）
probs = torch.softmax(outputs, dim=1)  # ❌ 不適用於 multi-label
```

### 5.2 Threshold Tuning（臨界值調整）

預設 threshold = 0.5，但通常不是最優：

```python
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, f1_score

def find_optimal_thresholds(y_true, y_pred_proba):
    """對每個類別找 F1-optimal threshold"""
    optimal_thresholds = []
    for i in range(y_true.shape[1]):
        precision, recall, thresholds = precision_recall_curve(
            y_true[:, i], y_pred_proba[:, i]
        )
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])  # 最後一個是 threshold=1
        optimal_thresholds.append(thresholds[best_idx])
    return np.array(optimal_thresholds)

# 使用方式
optimal_thresh = find_optimal_thresholds(y_val, val_preds)
predictions = (val_preds > optimal_thresh).astype(int)
```

### 5.3 多標籤評估指標

```python
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    average_precision_score
)

def compute_all_metrics(y_true, y_pred, y_proba, class_names):
    metrics = {}
    for i, name in enumerate(class_names):
        try:
            metrics[f'{name}_AUC'] = roc_auc_score(
                y_true[:, i], y_proba[:, i]
            )
        except ValueError:
            metrics[f'{name}_AUC'] = float('nan')

    # 整體 Macro / Micro 平均
    metrics['Macro_AUC'] = np.nanmean(list(metrics.values()))
    metrics['Macro_F1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['Micro_F1'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
    metrics['Example_Based_F1'] = f1_score(y_true, y_pred, average='samples', zero_division=0)
    return metrics
```

**CheXpert 論文的標準指標：AUROC（每個類別 + 平均）**

---

## 6. PyTorch 完整訓練程式碼框架

### 6.1 完整訓練腳本（可運行）

```python
#!/usr/bin/env python3
"""
胸腔 X 光 Multi-label 分類：完整訓練框架
使用 DenseNet-121 + CheXpert 資料集
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ 設定 ============
class Config:
    DATA_DIR = '/path/to/CheXpert-v1.0'
    IMAGE_DIR = '/path/to/CheXpert-v1.0/train'
    TRAIN_CSV = '/path/to/CheXpert-v1.0/train.csv'
    VAL_CSV = '/path/to/CheXpert-v1.0/valid.csv'  # 或自行切割
    OUTPUT_DIR = './outputs'
    MODEL_NAME = 'densenet121'  # or 'efficientnet_b0', 'vit_b_16'
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_EPOCHS = 15
    LR_HEAD = 1e-3
    LR_BACKBONE = 1e-5
    WEIGHT_DECAY = 0.01
    NUM_WORKERS = 4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LABEL_COLUMNS = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]

cfg = Config()
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ============ 資料集 ============
class ChestXrayDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, uncertain_as_positive=False):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.uncertain_as_positive = uncertain_as_positive
        self.label_cols = cfg.LABEL_COLUMNS

        # 填補缺失值
        for col in self.label_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(0.0)
                if uncertain_as_positive:
                    self.df[col] = self.df[col].replace(-1.0, 1.0)
                else:
                    self.df[col] = self.df[col].replace(-1.0, 0.0)

        # 移除 No Finding（或其他可選處理）
        self.df = self.df[self.df['Path'].notna()].reset_index(drop=True)
        logger.info(f'Loaded {len(self.df)} samples from {csv_path}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['Path'])

        # 讀取影像（轉灰階為 RGB）
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        # 標籤
        labels = torch.tensor(
            row[self.label_cols].values.astype(np.float32)
        )
        return img, labels


# ============ 模型 ============
class ChestXrayModel(nn.Module):
    def __init__(self, model_name='densenet121', num_classes=14, pretrained=True):
        super().__init__()
        if model_name == 'densenet121':
            self.backbone = models.densenet121(
                weights='IMAGENET1K_V1' if pretrained else None
            )
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()

        elif model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(
                weights='IMAGENET1K_V1' if pretrained else None
            )
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(
                weights='IMAGENET1K_V1' if pretrained else None
            )
            num_features = self.backbone.heads.head.in_features
            self.backbone.heads = nn.Identity()

        # Multi-label 分類頭（不要 Softmax，要 Sigmoid）
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ============ Transforms ============
train_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ============ 訓練函式 ============
def train_one_epoch(model, loader, optimizer, criterion, device, accum_steps=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc='Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels) / accum_steps
        loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accum_steps
        pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(loader, desc='Validating'):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 計算 AUROC
    aucs = []
    for i in range(all_labels.shape[1]):
        if all_labels[:, i].sum() > 0 and all_labels[:, i].sum() < len(all_labels[:, i]):
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            aucs.append(auc)

    mean_auc = np.mean(aucs) if aucs else 0.0
    return total_loss / len(loader), mean_auc, aucs


# ============ Main ============
def main():
    logger.info(f'Device: {cfg.DEVICE}')

    # 資料
    train_dataset = ChestXrayDataset(cfg.TRAIN_CSV, cfg.IMAGE_DIR, train_transform)
    val_dataset = ChestXrayDataset(cfg.VAL_CSV, cfg.IMAGE_DIR, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    # 模型
    model = ChestXrayModel(cfg.MODEL_NAME, num_classes=14).to(cfg.DEVICE)

    # 分層 LR
    optimizer = optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': cfg.LR_BACKBONE},
        {'params': model.classifier.parameters(), 'lr': cfg.LR_HEAD},
    ], weight_decay=cfg.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.NUM_EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    # 混合精度（大幅省記憶體）
    scaler = torch.cuda.amp.GradScaler()

    best_auc = 0
    for epoch in range(cfg.NUM_EPOCHS):
        logger.info(f'\nEpoch {epoch+1}/{cfg.NUM_EPOCHS}')

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.DEVICE)
        val_loss, val_auc, per_class_auc = validate(model, val_loader, criterion, cfg.DEVICE)
        scheduler.step()

        logger.info(f'Train Loss: {train_loss:.4f}')
        logger.info(f'Val Loss: {val_loss:.4f}, Val AUROC: {val_auc:.4f}')
        for i, (name, auc) in enumerate(zip(cfg.LABEL_COLUMNS, per_class_auc)):
            logger.info(f'  {name}: {auc:.4f}')

        # 存最好模型
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'best_model.pth'))
            logger.info(f'✓ Saved best model (AUC: {best_auc:.4f})')

    logger.info(f'\nTraining complete. Best AUROC: {best_auc:.4f}')


if __name__ == '__main__':
    main()
```

### 6.2 推論腳本

```python
@torch.no_grad()
def predict(model, img_path, transform, device, threshold=0.5):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    logits = model(img_tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (probs > threshold).astype(int), probs


# 使用
model = ChestXrayModel('densenet121').to(DEVICE)
model.load_state_dict(torch.load('outputs/best_model.pth'))
pred, prob = predict(model, '/path/to/test.jpg', val_transform, DEVICE)
for name, p, val in zip(LABEL_COLUMNS, pred, prob):
    print(f'{name}: {"✓" if p else "✗"} (confidence: {val:.3f})')
```

---

## 7. GPU 需求與雲端訓練方案

### 7.1 硬體需求速查

```
GPU 等級判斷：

6GB  (RTX 1060 / Tesla T4)    → DenseNet-121 batch=8 OK，EfficientNet batch=8
12GB (RTX 3060 / RTX 2080 Ti) → DenseNet-121 batch=32，EfficientNet batch=16
16GB (A4000 / RTX 3090)       → DenseNet-121 batch=48，EfficientNet batch=32
24GB (RTX 4090)                → DenseNet-121 batch=64，EfficientNet batch=48
40GB (A100)                    → DenseNet-121 batch=128，ViT-B/16 batch=32
80GB (A100-SXM4-80GB)          → 任意設定，訓練 MIMIC 全 dataset
```

### 7.2 雲端訓練方案比較

| 平台 | GPU 選項 | 成本（大概） | 優點 | 缺點 |
|------|---------|------------|------|------|
| **Google Colab Pro** | T4, A100 | $10/月 吃到飽 | 最快上手、有 CheXpert notebook | 會踢人、資源不穩定 |
| **Colab Enterprise** | V100, A100 | 按量計費 | 穩定、整合 GCS | 需 GCP 帳號 |
| **Kaggle** | T4, P100 | 每週 30h free | 有 Dataset 下載、類似 Colab | 額度有限 |
| **AWS SageMaker** | 多種 GPU | 按使用計費 | 生產級、管線完整 | 複雜、費用高 |
| **GCP Vertex AI** | T4, V100, A100 | 按量計費 | 整合 BigQuery、Custom Job 簡單 | 需 GCP 熟悉 |
| **Lambda Labs** | RTX 6000 / A100 | $0.50/h 起 | 性價比高、速度快 | 需管理 VM |

### 7.3 Colab 最快上路

```
1. 開啟 https://colab.research.google.com
2. Runtime → Change runtime type → GPU: A100 or T4
3. 安裝套件：
   !pip install torch torchvision pydicom pandas scikit-learn tqdm
4. 從 Google Drive 掛載資料：
   from google.colab import drive
   drive.mount('/content/gdrive')
5. 複製訓練腳本，替換路徑，開跑！
```

### 7.4 Colab 省時技巧

```python
# 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 存資料在 Drive，模型也存回 Drive
OUTPUT_DIR = '/content/drive/MyDrive/chest_xray/outputs'

# 用 GPU！
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')  # 確認有 GPU

# 掛 T4 可以跑的設定
# （colab T4 記憶體 ~15GB）
BATCH_SIZE = 16  # 小一點更穩
IMG_SIZE = 224
```

---

## 8. 常見陷阱與除錯指南

### 8.1 過擬合（Overfitting）

**徵兆：** Training loss 持續下降，但 Validation loss 不降反升

**解決方案（依優先順序）：**

```python
# 1. 最有效：增加資料 / Data Augmentation（見 4.5 節）

# 2. 正則化
model = ChestXrayModel(...)
#  dropout（已在程式碼中 0.3）
#  weight decay（已在 AdamW 中設 0.01）
#  label smoothing（對 multi-label 效果有限）

# 3. Early Stopping
patience = 5
best_val_loss = float('inf')
counter = 0
for epoch in range(NUM_EPOCHS):
    # ... train ...
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), 'best.pth')
    else:
        counter += 1
        if counter >= patience:
            logger.info(f'Early stopping at epoch {epoch}')
            break

# 4. 減小模型
#     DenseNet-121 → DenseNet-169 或 EfficientNet-B0
#     ViT-B/16 → ViT-S/16

# 5. 減小輸入解析度（最後手段）
#     512 → 384 → 224（影響準確率）
```

### 8.2 資料洩漏（Data Leakage）

**最常見錯誤：以 image-level 切分 train/val，忘了同一個病人的多張影像**

```python
# 錯誤示範
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2)  # ❌ 同一病人可能在 train 和 val！

# 正確做法：以 patient 為單位切分
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, val_idx in gss.split(df, groups=df['PatientID']):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

# CheXpert 資料集用 Path 欄位可抽出 PatientID：
# Path 格式: CheXpert-v1.0/patient00001/study1/view1_frontal.jpg
df['PatientID'] = df['Path'].apply(lambda x: x.split('/')[1])
```

### 8.3 Label Noise（標籤雜訊）

CheXpert 資料集的 U（不確定）和空白欄位有爭議：

```python
# 四種常見策略：
# 1. 全部視為 0（保守）
df = df.replace(-1, 0)

# 2. U 視為 1（進取，paper 中常用）
df = df.replace(-1, 1)

# 3. U 隨機分配（randomized U）
import random
for col in LABEL_COLUMNS:
    mask = df[col] == -1
    df.loc[mask, col] = df.loc[mask, col].apply(lambda _: random.choice([0, 1]))

# 4. Loss weighting（對 U 的 samples 降權）
sample_weights = torch.ones(len(labels))
# 如果有 uncertain label，降其權重
```

### 8.4 Loss 變 NaN

```python
# 常見原因與解決：
# 1. Learning rate 太高
lr = 1e-4  # 降低

# 2. 輸入影像太暗/太亮（標準化問題）
#    → 確認 Normalize 使用正確的 mean/std
#    → 先視覺化檢查：plt.imshow(img)

# 3. 除以零或 log(0)
#    → Focal Loss 中加 epsilon
epsilon = 1e-7
loss = ... + epsilon

# 4. 混合精度訓練（最推薦的解法）
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 8.5 類別不平衡（Class Imbalance）

CheXpert 的類別不平衡很嚴重：

```
No Finding:     ~70%（超多）
Pneumonia:      ~3%（很少）
Pneumothorax:   ~1%（極少）
```

```python
# 方法 1：Weighted BCE
pos_weight = torch.tensor([
    (labels[:, i] == 0).sum() / max((labels[:, i] == 1).sum(), 1)
    for i in range(14)
]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# 方法 2：Focal Loss（推薦，見 4.1 節）

# 方法 3：Oversampling 少數類別（小心不要造成過擬合）
from torch.utils.data import WeightedRandomSampler
class_counts = labels.sum(axis=0).numpy()
weights = 1.0 / class_counts
sample_weights = [weights[labels[i] == 1].mean() for i in range(len(labels))]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
```

### 8.6 程式除錯 Checklist

```
□ GPU 有被偵測到嗎？ torch.cuda.is_available()
□ 影像路徑正確嗎？ os.path.exists(img_path)
□ CSV 讀取正確嗎？ df.head() 看一下
□ 資料 transform 後正常？ plt.imshow(img)
□ Loss 一開始是多少？ 應該 ~0.69 (=-ln(0.5)) 左右
□ 記憶體夠嗎？ nvidia-smi 看 VRAM
□ 學習率對嗎？ 用 GradScaler 確認梯度有在更新
□ 模型輸出 shape 正確？ (batch, 14)
```

---

## 9. 快速上路懶人包

### 最快 30 分鐘跑起來

```bash
# 1. 安裝環境
pip install torch torchvision pandas scikit-learn tqdm pillow

# 2. 下載 CheXpert 資料（假設已完成）

# 3. 把上面的完整訓練腳本存為 train.py

# 4. 替換這三個路徑
DATA_DIR = '/your/path/to/CheXpert-v1.0'
TRAIN_CSV = '/your/path/to/CheXpert-v1.0/train.csv'
VAL_CSV   = '/your/path/to/CheXpert-v1.0/valid.csv'  # 或用 val.csv

# 5. 開跑
python train.py

# 6. 觀看訓練（等待 1-3 小時）
# Validation AUROC > 0.80 = 成功！
```

### 預期結果（CheXpert）

| 模型 | 平均 AUROC | 訓練時間（RTX 3090） |
|------|-----------|-------------------|
| DenseNet-121 (baseline) | 0.84-0.86 | ~2 小時 |
| EfficientNet-B0 | 0.85-0.87 | ~1.5 小時 |
| DenseNet-121 (CheXpert pretrained) | 0.87-0.89 | ~2 小時 |
| EfficientNet-B3 | 0.87-0.89 | ~4 小時 |
| ViT-B/16 (w/ strong aug) | 0.85-0.88 | ~6 小時 |

### 常見錯誤速查

| 問題 | 快速解法 |
|------|---------|
| CUDA OOM | batch_size 砍半，或加 `accum_steps=4` |
| Loss = NaN | lr 降 10 倍，用 `torch.cuda.amp` |
| 驗證集 AUROC = 0.5 | 模型根本沒 train 起來，檢查 lr + data |
| Val loss > Train loss | 正常一點點，太多 = 過擬合 |
| GPU 沒被用到 | 確認 `model.to(DEVICE)` + `images.to(DEVICE)` |
| 影像讀不到 | 檢查路徑大小寫，Linux 嚴格區分 |

---

## 延伸閱讀

- **CheXpert Paper**: Irvin et al., "CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels", AAAI 2019
- **DenseNet-121 (CheXNet)**: Rajpurkar et al., "CheXNet: Radiologist-Level Pneumonia Detection", arXiv 2017
- **EfficientNet**: Tan et al., "EfficientNet: Rethinking Model Scaling", ICML 2019
- **MedCLIP**: Wang et al., "MedCLIP", arXiv 2022
- **BioViL**: Bannur et al., "Learning to Exploit Temporal Structure for Biomedical Vision-Language Models", WACV 2023
- **PyTorch Image Models (timm)**: https://github.com/huggingface/pytorch-image-models — 包含所有 modern 架構

---

*本指南由 3F小馬 整理，如有任何問題或需要補充，歡迎提出！*
