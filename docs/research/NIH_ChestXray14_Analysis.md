# NIH ChestX-ray14 資料集分析報告

> Wang et al., CVPR 2017 (arXiv:1705.02315)  
> 原名 ChestX-ray8，後擴展為 14 標籤版本，由 CheXNet (Rajpurkar et al. 2017) 推廣為 ChestX-ray14

---

## 1. 資料集規模

| 指標 | 數值 |
|------|------|
| 總影像數 | 108,948 張正面 X 光片 |
| 獨立病患數 | 32,717 |
| 時間跨度 | 1992–2015（23 年） |
| 資料來源 | NIH Clinical Center PACS |
| 多標籤 | 每張影像可有 0~N 個病理標籤 |
| 有病理影像 | 24,636（22.6%）|
| 正常影像 | 84,312（77.4%）|
| Bounding box 標註 | 983 張影像，1,600 個框（每類 200）|

---

## 2. 14 個胸腔疾病標籤

### 原始 8 標籤（Wang et al. 2017）

| # | 英文名稱 | 中文 | ICD-10 | 定義 |
|---|---------|------|--------|------|
| 1 | **Atelectasis** | 肺擴張不全 | J98.11 | 肺部部分或完全塌陷 |
| 2 | **Cardiomegaly** | 心臟肥大 | I51.7 | 心臟增大 |
| 3 | **Effusion** | 胸腔積液 | J91 | 肋膜腔積液 |
| 4 | **Infiltration** | 浸潤 | R91.8 | 肺部異常物質（非特異性，已過時術語）|
| 5 | **Mass** | 腫塊 | R91.1 | X 光可見異常腫塊 |
| 6 | **Nodule** | 結節 | R91.1 | 小圓形陰影 |
| 7 | **Pneumonia** | 肺炎 | J18.9 | 肺部感染 |
| 8 | **Pneumothorax** | 氣胸 | J93 | 肺塌陷（空氣漏入胸腔）|

### 擴展 6 標籤（14-label version）

| # | 英文名稱 | 中文 | ICD-10 | 定義 |
|---|---------|------|--------|------|
| 9 | **Emphysema** | 肺氣腫 | J43 | 肺泡過度膨脹 |
| 10 | **Fibrosis** | 肺纖維化 | J84.10 | 肺部結疤 |
| 11 | **Pleural_Thickening** | 肋膜增厚 | J94.8 | 肋膜變厚 |
| 12 | **Consolidation** | 肺實質化 | J18.1 | 肺組織充滿液體/膿 |
| 13 | **Edema** | 肺水腫 | J81 | 肺水腫 |
| 14 | **Hernia** | 橫膈疝氣 | K44.9 | 橫膈疝氣（X 光可見）|

> 標籤來源：從放射報告以 NLP 文本挖掘（SNOMED-CT + UMLS Metathesaurus），非人工影像標註

---

## 3. 資料下載流程

### 下載來源

| 來源 | URL | 備註 |
|------|-----|------|
| Kaggle（推薦）| `kaggle.com/datasets/nih-chest-xrays` | 免費，需帳號 |
| TFDS | `tfds.load("nih_chest_x_ray")` | TensorFlow Datasets 內建 |
| HuggingFace | 社群副本（部分需申請）| 多個版本 |

### 檔案結構

```
nih-chest-xrays/
├── images/                          # 108,948 張 PNG 影像 (~7 GB)
├── Data_Entry_2017.csv              # 主要 metadata
├── Data_Entry_2017_v2020.csv        # 更新版（14 標籤完整版）
└── BBox_List_2017.csv               # Bounding box 標註（983 張）
```

### CSV 欄位 (Data_Entry_2017.csv)

| 欄位 | 說明 |
|------|------|
| Image Index | 檔名（如 00000001_000.png）|
| Finding Labels | 管線分隔的疾病標籤（如 "Atelectasis\|Effusion" 或 "No Finding"）|
| Follow-up # | 追蹤編號 |
| Patient ID | 病患 ID |
| Patient Age | 病患年齡 |
| Patient Gender | M/F |
| View Position | AP（前後）/ PA（後前）|
| OriginalImage[Width|Height] | 原始 DICOM 尺寸 |
| OriginalImagePixelSpacing[x|y] | 像素間距 |

### 下載指令

```bash
# Kaggle
pip install kaggle
kaggle datasets download -d nih-chest-xrays -p ./nih-chest-xrays/
unzip nih-chest-xrays.zip -d nih-chest-xrays/

# TFDS
import tensorflow_datasets as tfds
ds = tfds.load("nih_chest_x_ray", split="train", as_supervised=True)
```

---

## 4. 影像格式與解析度

| 屬性 | 值 |
|------|-----|
| 來源格式 | DICOM（醫院 PACS）|
| 分配格式 | PNG |
| 色彩空間 | 灰階（1 channel, 8-bit）|
| 分配解析度 | 1024 × 1024（從 DICOM 縮放）|
| 原始 DICOM 解析度 | 通常 ~2000 × 3000（因年代/設備而異）|
| 強度值處理 | 使用 DICOM header 預設 window level 轉換 |

> 注意：這是 X 光片，非 CT，**不適用 Hounsfield Unit window**

---

## 5. 標籤分佈（資料不平衡）

### 14 標籤正例統計（估計值，來自 v2020 CSV）

| 標籤 | 正例數 | 正例率 | 嚴重程度 |
|------|--------|--------|---------|
| Infiltration | ~19,810 | ~18.2% | 常見 |
| Effusion | ~13,317 | ~12.2% | 常見 |
| Atelectasis | ~11,559 | ~10.6% | 常見 |
| No Finding | ~84,312 | ~77.4% | 正常 |
| Nodule | ~6,291 | ~5.8% | 中等 |
| Mass | ~5,782 | ~5.3% | 中等 |
| Pneumothorax | ~5,302 | ~4.9% | 中等 |
| Consolidation | ~4,667 | ~4.3% | 中等 |
| Edema | ~2,305 | ~2.1% | 少見 |
| Emphysema | ~2,301 | ~2.1% | 少見 |
| Fibrosis | ~1,686 | ~1.5% | 少見 |
| Pleural Thickening | ~3,305 | ~3.0% | 中等 |
| Cardiomegaly | ~2,776 | ~2.5% | 少見 |
| Pneumonia | ~1,431 | ~1.3% | 稀有 |
| Hernia | ~227 | ~0.2% | 極稀有 |

> 正常 vs 任意疾病：~77:23  
> 最常見 vs 最稀有：Infiltration / Hernia ≈ **100:1**

---

## 6. 已知資料品質問題

### ⚠️ 標籤噪音
- 標籤由 NLP 從放射報告文本挖掘，非人工影像標註
- 整體 F1 ≈ 0.90，但 Mass recall 僅 0.40（一半漏掉）
- "Infiltration" 尤其不可靠，多篇論文建議移除

### ⚠️ Train/Test 資料洩漏
- 官方 70/10/20 split 以**影像**為單位（非病患）
- 同一病患的影像會同時出現在 train 和 test 中
- → 多篇論文的 AUC 成績被嚴重膨脹

### ⚠️ 不確定標籤處理
- 報告中的 "possible"、"suggestive of" 被當作負例
- 產生假陰性。CheXpert (Irvin et al. 2019) 正是為修復此問題而建

### ⚠️ 人群偏差
- 單一醫院（NIH Clinical Center，轉診中心），不代表全人口
- Zech et al. 證明此資料集訓練的模型在其他醫院失效

### ⚠️ AP vs PA 視角混合
- AP view 讓心臟顯得更大（影響 Cardiomegaly）
- 並非所有分析都控制此變數

---

## 7. 建議的取樣策略

| 策略 | 適用場景 | 實作方式 |
|------|---------|---------|
| **Weighted BCE** | 最常用 | 權重 = 1 / class_freq，或 median_freq / class_freq |
| **Focal Loss** | 稀有類別 | γ=2, α 按頻率加權 (Lin et al. 2017) |
| **Oversample 稀有類** | Hernia, Pneumonia | 每個 mini-batch 保證稀有類樣本 |
| **Undersample Normal** | 減少訓練時間 | Normal 從 ~60K 降到 ~10-20K |
| **兩階段訓練** | 效果最佳 | Stage 1: 全量訓練 → Stage 2: 平衡子集微調 |
| **Per-class threshold** | 推論時 | 每類獨立調最佳 threshold（非預設 0.5）|

**推薦組合：** Weighted BCE + Oversample 稀有類 + Per-class threshold tuning

---

## 8. 前處理建議

```python
# 推薦前處理 pipeline
PREPROCESSING = {
    # 1. 讀取 PNG (grayscale, 1 channel, 8-bit, 1024x1024)
    "load": {"format": "PNG", "color_mode": "grayscale"},
    
    # 2. Resize（配合 ImageNet backbone）
    "resize": 224,  # 或 256, 512 視需求
    
    # 3. Normalize（ImageNet mean/std）
    #    先 repeat 灰階 → 3 channels，再 normalize
    "normalize": {
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    },
    
    # 4. Data Augmentation（訓練時）
    "augmentation": {
        "random_horizontal_flip": 0.5,
        "random_rotation": 10,          # ±10°
        "random_translation": 0.1,      # ±10%
        "random_scale": (0.9, 1.1),
        "brightness_jitter": 0.2,
        "contrast_jitter": 0.2,
    },
}
```

### 關鍵注意事項

1. **不要用 Hounsfield window** — 這是 X 光片不是 CT
2. **灰階 → 3 channels** — `x.repeat(3, axis=-1)` 或 `torch.cat([x]*3, dim=0)`
3. **AP/PA 分開處理（可選）** — 如果要控制 Cardiomegaly 偏差
4. **CLAHE（可選）** — 對比度增強，部分研究有效

---

## 9. Train/Val/Test 分割建議

### ⚠️ 必須：Patient-level split

```python
from sklearn.model_selection import GroupShuffleSplit

# 按 Patient ID 分組，避免資料洩漏
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df["Patient ID"]))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
# 0.125 × 0.8 = 0.1 → final: 70/10/20
sub_train, val_idx = next(gss2.split(df.iloc[train_idx], 
                                       groups=df.iloc[train_idx]["Patient ID"]))
val_idx = train_idx[sub_train]
train_idx = train_idx[sub_test]  # adjusted indices
```

### 分割比

| Split | 比例 | 依據 |
|-------|------|------|
| Train | 70% | by Patient ID |
| Val   | 10% | by Patient ID |
| Test  | 20% | by Patient ID |

---

## 10. 參考資料

| 文獻 | 年份 | 關鍵貢獻 |
|------|------|---------|
| Wang et al., "ChestX-ray8" | 2017 CVPR | 原始資料集 + 8 標籤基線 |
| Rajpurkar et al., "CheXNet" | 2017 | 121-layer DenseNet, 推廣 14 標籤 |
| Irvin et al., "CheXpert" | 2019 | 修正不確定標籤處理 |
| Zech et al. | 2018 | 跨域泛化失敗 |

---

## 附錄：快速 CSV 分析指令

```python
import pandas as pd

df = pd.read_csv("Data_Entry_2017_v2020.csv")
print(f"Total images: {len(df)}")
print(f"Unique patients: {df['Patient ID'].nunique()}")
print(f"\nLabel distribution:")
labels = df["Finding Labels"].str.split("|").explode()
print(labels.value_counts())
print(f"\nLabels per image:")
print(df["Finding Labels"].str.split("|").str.len().describe())
print(f"\nView Position:")
print(df["View Position"].value_counts())
```
