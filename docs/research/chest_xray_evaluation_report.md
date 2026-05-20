# 胸腔 X 光多標籤分類評估指標與最佳實踐報告

**目標對象：** Reviewer 階段驗收標準制定  
**覆蓋範圍：** 1. 評估指標  2. 多標籤陷阱  3. 類別不平衡策略  4. Benchmark 數據  5. 倫理考量  6. 驗證策略  7. 可解釋性工具整合  

---

## 1. 評估指標：核心指標詳解

### 1.1 AUC-ROC（首選指標）

**為何是胸腔 X 光事實標準：**

- CheXpert 官方比賽以 **AUC-ROC per class** 作為 primary metric（CheXpert: Large Dataset for Chest X-ray Interpretation, Irvin et al., 2019）
- AUC 不受閾值選擇影響，適合類別不平衡的醫學影像場景
- CheXNet（Rajpurkar et al., 2017）開創以來，所有主流 CXR 論文均以 AUC 報告

**計算方式（多標籤情境）：**

每個類別獨立做 One-vs-Rest（OvR）二分類：
1. 對類別 $c$，取所有影像在該類別的預測機率 vs 標籤
2. 計算該類別的 AUC-ROC
3. 最終報告 14 個類別各自的 AUC，最後取 Mean AUC

```python
# sklearn 實作
from sklearn.metrics import roc_auc_score
for cls in classes:
    y_true_cls = (labels == cls)   # 二值化
    y_score_cls = probs[:, cls]    # 預測機率
    auc = roc_auc_score(y_true_cls, y_score_cls)
```

**解讀標準：**
| AUC 範圍 | 臨床意義 |
|----------|----------|
| 0.90–1.0 | Excellent — 接近或超越住院醫師 |
| 0.80–0.90 | Good — 可作為輔助工具 |
| 0.70–0.80 | Fair — 需搭配其他指標謹慎解讀 |
| < 0.70 | Poor — 不適合臨床應用 |

### 1.2 Macro / Micro / Weighted F1

| 指標 | 計算方式 | 適用場景 |
|------|----------|----------|
| **Micro F1** | 全域 TP / (TP + 0.5 FP + FN) | 重視多數類別（常見疾病） |
| **Macro F1** | 每類 F1 的算術平均 | 平等對待所有類別（類別數量相等時） |
| **Weighted F1** | 每類 F1 加權（按類別樣本數） | 真實標籤分佈下的整體表現 |

**在 CXR 的特殊考量：**
- Micro F1 會被高頻類別（如"No Finding"）主導，容易高估少數類別（如"Pneumothorax"）的表現
- **Macro F1 是多標籤不平衡分類的首選 F1 指標**，但需注意類別數量權重問題
- 建議同時報告 Macro F1 和 Weighted F1，並明確說明閾值選擇策略

**F1 的根本限制：** F1 是閾值依賴的（threshold-dependent），不同閾值會產生截然不同的 F1。**務必報告閾值選擇方式**（固定 0.5、Youden's J、最大 F1、或臨床指定）。

### 1.3 Precision-Recall 曲線

**何時用 PR 而非 ROC：**

- 類別**極度不平衡**時（正樣本 < 5%），ROC 會過度樂觀（曲線被人為拉高）
- PR-AUC 能真實反映少數類別的檢測能力

**胸腔 X 光常見低發生率類別（發生率 < 5%）：**
- Pneumothorax: ~1–2%
- Pneumoperitoneum: ~0.5%
- Emphysema: ~1%

```python
from sklearn.metrics import precision_recall_curve, auc
pr_auc = auc(precision, recall)  # PR-AUC
```

**建議：** 所有類別均報告 PR-AUC，特別是發生率 < 5% 的類別僅用 PR-AUC 解讀。

### 1.4 完整報告清單（最低要求）

```
必報：
  - AUC-ROC per class（14 classes）
  - Mean AUC-ROC（across all classes）
  - Macro F1 @ optimal threshold
  - Micro F1 @ optimal threshold
  - PR-AUC per class（針對發生率 < 10% 的類別）

強烈建議：
  - AUC-ROC 95% confidence interval（bootstrap 或 DeLong test）
  - 閾值選擇策略說明
  - 類別發生率（prevalence）表格
```

---

## 2. 多標籤分類評估陷阱

### 2.1 Subset Accuracy（精確匹配）為何不適用

Subset Accuracy = 所有標籤完全正確才算正確

$$\text{Subset Accuracy} = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[y_i = \hat{y}_i]$$

**根本問題：**

1. **指數爆炸**：14 類分類就有 16,384 種可能組合，任一類預測錯誤即判為失敗
2. **隨機基線極低**：若平均每張影像有 2.5 個正標籤，隨機猜測 Subset Accuracy ≈ $0.5^{14} \approx 0.006\%$（實際因標籤相關性略高，但仍 < 1%）
3. **臨床無意義**：臨床情境從不要求「所有14類完全正確」，而是「關鍵疾病正確檢出」

**文獻共識：** Subset Accuracy 在多標籤文獻中被廣泛棄用。請在 Method 章節明確說明不使用 Subset Accuracy 並陳述理由。

### 2.2 Hamming Loss

$$\text{Hamming Loss} = \frac{1}{N \cdot L}\sum_{i=1}^N \sum_{l=1}^L \mathbb{1}[y_{i,l} \neq \hat{y}_{i,l}]$$

- 衡量「每個標籤位」的錯誤率
- 優點：解讀直觀、計算簡單
- 缺點：**對類別不平衡不敏感**，無法區分「漏報」和「誤報」的成本差異
- 建議：作為輔助指標報告，但不作為 primary metric

### 2.3 標籤相依性問題

胸腔 X 光中標籤高度相關：
- "Cardiomegaly" 常與 "Enlarged Cardiomediastinum" 共現
- "Pneumonia" 常與 "Consolidation" 共現
- "Pneumothorax" 幾乎不與 "Pleural Effusion" 共現

**評估陷阱：** 若只用 per-class AUC 會忽略標籤相依性。建議：
- 報告 Label Co-occurrence Matrix（標籤共現矩陣）
- 考慮報告 **Example-Based** 指標（Hamming Loss、Subset Accuracy 之外）如 Ranking Loss 或 Coverage

### 2.4 不一致標籤（Label Noise）

CheXpert 資料集的標籤來自 CheXbert NLP 模型（非完美）：
- "Uncertainty" 標籤（如 U-Zeros、U-Ones 策略）的處理方式會影響評估
- MIMIC-CXR 的標籤同樣來自 NLP extraction，有類似問題

**建議：** 在 Results 章節報告：
1. 使用的標籤不確定性策略（remove uncertain / treat as positive / treat as negative）
2. 考慮用 Cohen's Kappa 評估標註者一致性

---

## 3. 類別不平衡下的評估策略

### 3.1 類別不平衡程度（CheXpert 14 類）

| 類別 | 發生率（典型） |
|------|----------------|
| No Finding | 30–50% |
| Cardiomegaly | 15–25% |
| Pleural Effusion | 15–20% |
| Lung Lesion | 3–5% |
| Pneumonia | 3–5% |
| Pneumothorax | 1–2% |
| Fracture | 1–2% |

不平衡比例（majority/minority）可達 **50:1**。

### 3.2 評估時的對應策略

**策略一：報告分層指標**
- Per-class precision、recall、specificity 分別報告
- 分層抽樣確保測試集中各類別比例與真實分佈一致

**策略二：AUC + PR-AUC 雙軌制**
- AUC 反映整體排序能力（不受閾值影響）
- PR-AUC 反映少數類別的檢測代價
- 兩者共同報告才能完整描述模型能力

**策略三：臨床驅動閾值**
- 固定 0.5 閾值對極少數類別（如 Pneumothorax）無意義
- 臨床情境應根據**敏感度需求**設定閾值：
  - 危急疾病（pneumothorax）：要求 sensitivity ≥ 95% → 低閾值、高 recall
  - 非危急疾病（emphysema）：要求 specificity ≥ 90% → 高閾值、高 precision
- 建議報告 **recall@95% sensitivity** 和 **specificity@90% sensitivity**（臨床可解讀的指標）

**策略四：校正後預測（Calibrated Probability）**
- 檢查 predicted probability 的 Calibration： reliability diagram、Brier Score
- Platt Scaling 或 Isotonic Regression 後再評估 F1

---

## 4. Benchmark 數據供對比

### 4.1 CheXpert 資料集

- **規模：** 224,316 張胸腔 X 光，65,240 位患者（Stanford Hospital）
- **標籤：** 14 個類別（CheXpert labeler + CheXbert NLP extraction）
- **評估設定：** 官方 5 類（Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion）

**官方 Baseline（DenseNet-121）14 類 Mean AUC = 0.889**

| 類別 | U-Ones Mean AUC |
|------|----------------|
| Atelectasis | 0.893 |
| Cardiomegaly | 0.904 |
| Consolidation | 0.889 |
| Edema | 0.941 |
| Pleural Effusion | 0.932 |
| Lung Opacity | 0.896 |
| Pneumonia | 0.803 |
| Pneumothorax | 0.888 |
| **Mean（5-class official）** | **0.903** |
| Pleural Other | 0.717 |
| Fracture | 0.823 |
| No Finding | 0.964 |

### 4.2 MIMIC-CXR 資料集

- **規模：** 377,110 張影像，227,835 份研究，BIDMC（2011-2016）
- **標籤：** 13 個類別（CheXbert NLP extraction）
- **難度：** 通常比 CheXpert 更具挑戰性（跨機構泛化）

**Baseline Performance（MIMIC-CXR）：**
| Model | Mean AUC |
|-------|----------|
| ResNet-50 | ~0.87–0.89 |
| DenseNet-121 | ~0.88–0.91 |
| BioViL zero-shot | 0.842 |
| RETFound fine-tuned | ~0.87–0.90 |
| ViT-Large (2024) | >0.92 |

### 4.3 橫跨機構泛化（Cross-Dataset Generalization）

| 訓練集 → 測試集 | AUC 降幅 |
|----------------|----------|
| CheXpert → MIMIC-CXR | -3% to -8% mean AUC |
| MIMIC-CXR → CheXpert | -5% to -10% mean AUC |

**對 Reviewer 的啟示：**
- 需報告 internal validation（in-distribution）和 external validation（out-of-distribution）兩種 AUC
- 若聲稱跨機構泛化能力，**必須有外部資料集測試結果**

### 4.4 SoTA 對照表（CheXpert 5-Class Mean AUC）

| Year | Method | Mean AUC |
|------|--------|----------|
| 2019 | DenseNet-121 supervised (U-Ones) | 0.903 |
| 2019 | Competition ensemble winner | 0.927 |
| 2022 | CheXzero (zero-shot CLIP) | 0.936 |
| 2022 | BioViL (zero-shot VLM) | 0.950 |
| 2024 | CheXagent (fine-tuned 8.5B) | ~0.95+ |
| 2024 | BiomedCLIP (zero-shot) | ~0.93–0.95 |

---

## 5. 醫學影像 AI 的倫理考量

### 5.1 偏見問題（Bias）

**文獻記錄的偏見類型：**

1. **人口統計偏見（Demographic Bias）**
   - Larrazabal et al. (2020)：性別不平衡導致女性病患類別召回率顯著低於男性
   - 種族偏見：黑人及西班牙裔患者在 ARDS、COVID-19 肺部表現在不同機構中表現更差
   - 年齡偏見：高齡患者（>75歲）影像因拍攝品質差異模型表現下降

2. **機構偏見（Institutional Bias）**
   - 訓練資料集中於特定醫院系統，模型對其他品牌設備、不同體位（AP vs PA）表現退化
   - 跨機構測試時 AUC 通常下降 3–10%

3. **標籤偏見（Label Bias）**
   - CheXpert 標籤由 NLP 模型自動生成，存在 systematic error pattern
   - No Finding 類別與其他類別互斥假設不一定成立

**緩解策略：**
- 報告性別/種族/年齡分層的 AUC（subgroup AUC）
- 計算 **Equalized Odds Difference** 或 **Demographic Parity Gap**
- 進行 Fairness Audit：確保各人口統計子群體的 sensitivity 和 specificity 差異 < 5%

### 5.2 可解釋性要求

- Grad-CAM 等視覺化已成為 FDA 審查標配要求
- TRIPOD-AI（2024 update）要求 AI 模型的預測解釋納入報告
- **禁止「黑箱預測」用於臨床決策**：必須提供人類可理解的解釋

### 5.3 臨床驗證台階

| 階段 | 驗證類型 | 最低要求 |
|------|----------|----------|
| 開發階段 | Internal validation（ retrospective） | Patient-level stratified split, 5-fold CV |
| 轉化階段 | External validation | 至少一個外部資料集/機構 |
| 臨床前期 | Prospective study | IRB 批准，多中心 |
| 臨床階段 | Randomized controlled trial | 對照組設計 |
| 監管階段 | FDA 510(k) / De Novo | 安全性與有效性數據 |

---

## 6. 建議的驗證策略

### 6.1 Patient-Level Split（關鍵！）

**最常見的方法論錯誤：** 圖像層級（image-level）隨機分割 → 同一患者出現在 train/test → **資料洩漏** → AUC 人為提高 5–15%

**正確做法：** 以患者為單位分割（patient-level split）

```python
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=patient_ids))
# 確保同一患者的所有影像在同一 split
```

### 6.2 分層抽樣（Stratified Split）

多標籤情境下，需同時按**多個類別**分層：
- 使用 MultilabelStratifiedKFold（iterative stratification 演算法）
- 確保每個 split 中各類別的正樣本比例一致

```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y_multilabel)):
    ...
```

### 6.3 Cross-Validation 架構建議

| 方法 | 說明 | 適用場景 |
|------|------|----------|
| 5-fold CV | 標準推薦 | 資料量中等（> 10,000） |
| 10-fold CV | 更穩定估計 | 需要高可靠度時 |
| Repeated 5-fold | 減少 variance | 小資料集 (< 5,000) |
| Leave-One-Site-Out | 跨機構泛化 | 多中心研究 |

### 6.4 測試集要求

1. **測試集大小：** 至少 500–1,000 張（CheXpert 官方比賽用 200 個報告樣本）
2. **不重複揭露測試集：** 嚴禁在測試集上調參
3. **預先註冊（Pre-registration）：** 建議在 Open Science Framework 預先註冊分析計畫，防止 p-hacking

---

## 7. 可解釋性工具整合建議

### 7.1 Grad-CAM（最廣泛使用）

**原理：** 計算最終卷積層的梯度加權 Activation Map

```python
# PyTorch 實作框架
model = models.densenet121(pretrained=True)
model.eval()
target_layer = model.features.denseblock4.denselayer16
target_activations = []
gradients = []

def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0])

def forward_hook(module, input, output):
    target_activations.append(output)

handle1 = target_layer.register_forward_hook(forward_hook)
handle2 = target_layer.register_full_backward_hook(backward_hook)

# ... forward + backward pass ...

grad_cam = (target_activations[0] * gradients[0].mean(dim=(2,3), keepdim=True)).sum(dim=1)
```

**在 CXR 的使用建議：**
- 報告每個類別的代表性 Grad-CAM 視覺化
- 進行 **pointing game**：在影像上標註 disease location，驗證 Grad-CAM 熱區是否落在疾病區域
- **不要**只靠 Grad-CAM 視覺化作為解釋，需量化（如 bounding box overlap IoU）

### 7.2 Attention Maps（ViT / Transformer 架構）

- Self-attention maps 直接可視化關注區域
- CLS token attention 作為全局摘要
- **建議量化：** attention rollout 技術（Abnar & Zuidema, 2020）展現多層 attention 疊加效果

### 7.3 SHAP（Model-Agnostic）

- 計算每個像素的 SHAP value
- 缺點：計算代價極高（全圖素 x 訓練樣本），需 subsampling
- 適用場景：非 CNN 模型（Random Forest、Logistic Regression）或模型不可微分時

### 7.4 多方法 XAI 整合策略

| 工具 | 適用架構 | 計算成本 | 解讀難易度 |
|------|----------|----------|------------|
| Grad-CAM | CNN | 低 | 中 |
| Grad-CAM++ | CNN | 中 | 中 |
| Attention Maps | ViT/Transformer | 低 | 中 |
| SHAP | 任意 | 高 | 高 |
| LIME | 任意 | 中 | 高 |

**建議的最低 XAI 報告要求：**
1. 每個類別取 3 個代表性案例的 Grad-CAM
2. 量化熱區覆蓋率（IoU vs radiologist-annotated bounding box）
3. 專家評估：讓放射科醫師評估 Grad-CAM 是否指向合理區域

---

## 8. Reviewer 驗收標準（摘要）

以下是讓 Reviewer 可直接執行的驗收檢查清單：

### 8.1 Primary Metrics（必須達標）

```
□ Mean AUC-ROC ≥ 0.90（in-distribution, CheXpert-style 14類）
□ 每類 AUC-ROC 報告完整表格
□ Macro F1 @ optimal threshold 報告
□ PR-AUC 報告（發生率 < 10% 的類別）
□ 類別發生率表格（Prevalence table）
□ 閾值選擇策略說明
```

### 8.2 Methodology（方法論合規）

```
□ Patient-level split（嚴禁 image-level split）
□ MultilabelStratifiedKFold 或等效分層策略
□ 95% CI（bootstrap 或 DeLong test for AUC）
□ 明確說明標籤不確定性策略（U-Ones / U-Zeros / remove）
□ 預訓練模型來源與訓練配置描述
```

### 8.3 Fairness & Ethics（倫理合規）

```
□ 性別分層 AUC（Male vs Female subgroup AUC）
□ 種族分層 AUC（如資料可取得）
□ 標籤共現矩陣（Label co-occurrence matrix）
□ 臨床驗證階段說明（retrospective / prospective）
```

### 8.4 Interpretability（可解釋性）

```
□ Grad-CAM per class（至少 3 例/類別）
□ 熱區量化覆蓋率（IoU 或 pointing game）
□ XAI 視覺化範例含原始影像 + Grad-CAM overlay + 預測解讀
```

### 8.5 External Validation（如聲稱泛化能力）

```
□ 跨機構測試集 AUC（至少一個外部資料集）
□ 明確說明訓練集與外部測試集的機構差異
□ Cross-dataset drop 量化（AUC difference）
```

---

## 9. 參考文獻

1. Irvin J, et al. CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison. AAAI 2019.
2. Rajpurkar P, et al. CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. arXiv 2017.
3. Tiu E, et al. Zero-shot anomaly detection for medical images. Nature Communications 2022.
4. Zhou YH, et al. RETFound: Foundation model for medical imaging with diagnostic AI. Nature 2023.
5. Jain S, et al. CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation. Nature 2024.
6. Larrazabal AJ, et al. Gender imbalance in medical imaging datasets. Nat Med 2020.
7. CONSORT-AI / TRIPOD-AI reporting guidelines (2024 update)
8. FDA AI/ML Action Plan / PCCP Guidance (Dec 2024)
9. Tseng YH, et al. Bidirectional Learning for Domain Adaptation of Medical Imaging. TMI 2020.
10. Abnar S, Zuidema W. Quantifying Attention Flow in Transformers. ACL 2020.

---

*本報告為研究型文件，數值數據來自已發表文獻。特定 benchmark 數值建議在提交前對照最新 Stanford AIMI CheXpert Leaderboard 和 PapersWithCode 確認。*
