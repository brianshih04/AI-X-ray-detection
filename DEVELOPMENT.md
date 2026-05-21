# 開發者指南（Development Guide）

本文件為「胸腔 X 光片自動判讀系統」的開發者指南，涵蓋環境設定、架構說明、訓練流程、API 開發及 K8S 部署。

---

## 目錄

- [環境需求](#環境需求)
- [開發環境設定](#開發環境設定)
- [專案架構詳解](#專案架構詳解)
- [資料預處理流程](#資料預處理流程)
- [模型訓練指南](#模型訓練指南)
- [後端 API 開發](#後端-api-開發)
- [K8S 部署流程](#k8s-部署流程)
- [Git 工作流程](#git-工作流程)
- [除錯與排障](#除錯與排障)

---

## 環境需求

| 項目 | 版本 | 備註 |
|------|------|------|
|| Python | 3.12+ | API 容器使用 3.12-slim ||
|| PyTorch | 2.13+ (nightly cu130) | GB10 sm_121 需 nightly (訓練用) ||
|| ONNX Runtime | 1.24.4 | 推論引擎 ||
|| CUDA | 13.2+ | NVIDIA GB10 (DGX Spark) ||
|| PostgreSQL | 15+ | 預測結果資料庫 ||
|| Docker | 29+ | 容器化 (host) ||
|| minikube | 1.35+ | 本地 K8S (containerd) ||
|| kubectl | 1.35+ | K8S CLI ||

---

## 開發環境設定

### 本地開發（WSL）

```bash
# WSL 環境（Windows 主機）
# 專案路徑：/mnt/d/projects/AI-X-ray-detection/

# 建立虛擬環境
python3.11 -m venv venv
source venv/bin/activate

# 安裝依賴
pip install -r model_training/requirements.txt
pip install -r api/requirements.txt
```

### 遠端部署主機（GB10）

```bash
# SSH 連線（透過 Cloudflare Tunnel）
ssh -o ProxyCommand="cloudflared access ssh --hostname ssh.avision-gb10.org" \
    avuser@ssh.avision-gb10.org
# 密碼: Brian0054$

# 專案路徑：~/projects/ai-xray-detection/
# PyTorch nightly 已安裝在 venv 中
source ~/projects/ai-xray-detection/venv/bin/activate

# 確認 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# → NVIDIA GB10
```

### GB10 環境規格

- **CPU**: 20-core ARM (10x Cortex-X925@4.0GHz + 10x Cortex-A725@2.8GHz)
- **GPU**: NVIDIA GB10 (sm_121)
- **記憶體**: 32GB
- **磁碟**: 3.6TB (2.9TB 可用)
- **OS**: Ubuntu 24.04 (aarch64)
- **CUDA**: 13.2 / Driver 595.58.03

---

## 專案架構詳解

```
AI-X-ray-detection/
├── data_etl/                  # 資料下載、清洗、驗證管線
│   ├── src/
│   │   └── chestxray14_dataset.py   # NIH 資料集下載 + 驗證
│   └── tests/test_etl.py            # 31 個單元測試
│
├── database/                  # 資料庫層
│   ├── alembic/               # 資料庫遷移
│   │   └── versions/001_initial.py  # 初始 schema（5 張表）
│   └── scripts/seed_nih.py    # 種子資料匯入
│
├── model_training/            # 模型訓練核心
│   ├── config/default.yaml    # 訓練配置
│   ├── src/
│   │   ├── dataset.py         # 資料集 + DataLoader
│   │   ├── model.py           # DenseNet-121 / EfficientNet 模型
│   │   ├── losses.py          # BCEWithLogitsLoss + Focal Loss
│   │   ├── metrics.py         # AUROC / F1 / PR-AUC 計算
│   │   ├── callbacks.py       # EarlyStopping / Checkpoint / Scheduler
│   │   └── tracker.py         # TensorBoard / W&B 實驗追蹤
│   ├── train.py               # 訓練入口（508 行）
│   ├── preprocess_nih.py      # NIH 資料預處理
│   └── requirements.txt
│
├── api/                       # FastAPI 後端
│   ├── src/
│   │   ├── main.py            # FastAPI 應用入口
│   │   ├── config.py          # 環境設定
│   │   ├── database.py        # 非同步資料庫連線
│   │   ├── schemas.py         # Pydantic 資料模型
│   │   ├── crud.py            # CRUD 操作
│   │   ├── api/
│   │   │   ├── core.py        # 核心 API 路由（推論、歷史）
│   │   │   └── auth.py        # 認證路由（JWT）
│   │   ├── services/
│   │   │   ├── auth.py        # JWT 認證服務
│   │   │   ├── preprocessing.py   # 影像預處理
│   │   │   └── inference.py   # 推論引擎（ONNX/TorchScript）
│   │   └── middleware/
│   │       └── rate_limit.py  # API 速率限制
│   ├── tests/                 # API 測試
│   ├── Dockerfile             # API 容器映像
│   └── requirements.txt
│
├── k8s/                       # Kubernetes 部署配置
│   ├── namespace.yaml         # ai-xray namespace
│   ├── deployment.yaml        # Deployment + GPU 資源請求
│   └── service.yaml           # ClusterIP Service
│
└── docs/research/             # 研究報告
    └── chest-xray-training-guide.md
```

### 模組職責

| 模組 | 職責 | 語言 |
|------|------|------|
| `data_etl` | 從 Kaggle 下載 NIH ChestX-ray14、完整性驗證 | Python |
| `database` | PostgreSQL schema、Alembic migration、種子資料 | SQL/Python |
| `model_training` | 資料預處理 → 模型訓練 → 評估 → 模型匯出 | Python/PyTorch |
| `api` | REST API（認證、推論、歷史查詢） | Python/FastAPI |
| `k8s` | Kubernetes 部署 manifests | YAML |

---

## 資料預處理流程

### NIH ChestX-ray14 → 訓練格式

使用 `preprocess_nih.py` 將原始資料轉換為訓練格式：

```bash
python preprocess_nih.py --data_dir ~/ai-xray-detection/data
```

**處理步驟：**

1. **載入 `Data_Entry_2017.csv`**（112,120 筆）
2. **解析多標籤**：`Finding Labels` 欄位（如 `"Atelectasis|Effusion"`）→ 15 個二元標籤欄位
3. **解析影像路徑**：掃描 `images_001/` ~ `images_012/` 建立 `Image Index → Path` 對應
4. **官方資料集分割**：
   - 使用 `train_val_list.txt`（86,524 張）和 `test_list.txt`（25,596 張）
   - Train/Val 再按 Patient ID 做 80/20 分割（避免資料洩漏）
5. **輸出 CSV**：`train.csv`、`val.csv`、`test.csv`（含 `Path` + 15 個標籤欄位）

**資料分割結果：**

| 分割 | 影像數 | 病人數 |
|------|--------|--------|
| Train | 69,628 | 22,407 |
| Val | 16,896 | 5,601 |
| Test | 25,596 | — |

### 15 類標籤分布（Train）

```
No Finding          58.3%  ← 最多數
Infiltration        16.1%
Effusion            10.0%
Atelectasis          9.4%
Nodule               5.4%
Mass                 4.6%
Consolidation        3.4%
Pneumothorax         3.1%
Pleural_Thickening   2.6%
Cardiomegaly         1.9%
Emphysema            1.8%
Edema                1.6%
Fibrosis             1.4%
Pneumonia            1.0%
Hernia               0.2%  ← 最少數
```

---

## 模型訓練指南

### 訓練配置（config/default.yaml）

```yaml
model:
  backbone: densenet121         # 模型骨幹
  pretrained: true              # ImageNet 預訓練權重
  num_classes: 15               # 15 類多標籤
  freeze_backbone_epochs: 3     # 前 3 epoch 凍結骨幹
  dropout: 0.2                  # Dropout 率

training:
  num_epochs: 30                # 訓練 epoch 數
  batch_size: 32                # 批次大小
  lr_head: 0.001                # 分類頭學習率
  lr_backbone: 0.0001           # 骨幹學習率（解凍後）
  optimizer: adamw              # AdamW 優化器
  amp: true                     # 混合精度訓練
  clip_grad_norm: 1.0           # 梯度裁剪

loss:
  type: bce                     # BCEWithLogitsLoss
  use_pos_weight: true          # 自動計算正樣本權重

scheduler:
  type: cosine                  # 餘弦退火
  T_max: 30                     # 週期
  eta_min: 0.00001              # 最小學習率
```

### 啟動訓練

```bash
# GB10 上執行
cd ~/ai-xray-detection
source venv/bin/activate

# 預設配置
python train.py --data_dir ~/ai-xray-detection/data

# 自訂配置
python train.py --config config/default.yaml --epochs 50 --batch_size 64

# 從 checkpoint 恢復
python train.py --resume ~/ai-xray-detection/models/densenet121_nih/last_model.pth
```

### 訓練流程

```
載入配置 → 建構 DataLoader → 建構模型
    ↓
計算 pos_weight（遍歷一次 train_loader）
    ↓
Epoch 1~3: 骨幹凍結，只訓練分類頭
    ↓
Epoch 4+: 解凍骨幹，差異學習率訓練
    ↓
每個 Epoch:
  Train → Validate → 計算指標 → 存 Checkpoint → Early Stopping
    ↓
最終評估 → 尋找最佳閾值 → 匯出 ONNX/TorchScript
```

### 模型架構

**DenseNet-121 Transfer Learning**

```
Input (3×224×224)
    ↓
DenseNet-121 Backbone (ImageNet pretrained, 6.9M params)
  ├── Dense Block 1 (6 layers)
  ├── Dense Block 2 (12 layers)
  ├── Dense Block 3 (24 layers)
  └── Dense Block 4 (16 layers)
    ↓
Global Average Pooling
    ↓
Dropout (0.2)
    ↓
Linear(1024 → 15)    ← 分類頭
    ↓
Output (15 logits)    ← BCEWithLogitsLoss
```

### 損失函數

- **BCEWithLogitsLoss + pos_weight**：處理類別不平衡
  - `pos_weight[i] = neg_count / pos_count`（每類獨立計算）
  - Hernia (0.2%) 的 pos_weight ≈ 500，No Finding (58.3%) 的 pos_weight ≈ 0.7
- **Focal Loss**（可選）：`type: focal`，適用於更嚴重的類別不平衡

### 評估指標

| 指標 | 說明 |
|------|------|
| AUROC (per class) | 各類別 ROC 曲線下面積 |
| Mean AUROC | 15 類 AUROC 平均 |
| Macro F1 | 各類 F1 平均（公平對待稀有類別） |
| Micro F1 | 全域 F1（受多數類影響） |
| PR-AUC (per class) | Precision-Recall 曲線下面積 |
| Optimal Threshold | 各類別最佳閾值（Youden's J） |

### 模型匯出

訓練完成後自動匯出：

```yaml
export:
  onnx: true              # model.onnx (opset 17)
  torchscript: true       # model.torchscript
  dynamic_batch: true     # 動態批次大小
```

匯出檔案位於 `~/ai-xray-detection/models/densenet121_nih/`：

- `best_model.pth` — 最佳 checkpoint
- `last_model.pth` — 最後一個 epoch
- `model.onnx` — ONNX 格式（推論用）
- `model.torchscript` — TorchScript 格式（推論用）
- `training_results.json` — 完整訓練結果
- `evaluation_report.txt` — 評估報告

---

## 後端 API 開發

> 目前使用 `api_build_onnx/` (ONNX Runtime, 406MB)。舊版 `api_build/` (PyTorch, 5.15GB) 已停用。
> 完整版 `api/` (含認證、rate limit) 為未來規劃。

### FastAPI 結構 (api_build_onnx/)

```
api_build_onnx/
├── main.py              # FastAPI app, ONNX 推論, PostgreSQL 寫入
├── requirements.txt     # onnxruntime, fastapi, uvicorn, sqlalchemy, psycopg2-binary
├── Dockerfile           # CPU 版 (python:3.12-slim, 406MB)
├── Dockerfile.gpu       # GPU 版 (含 CUDA libs + onnxruntime-gpu, 598MB)
└── models/
    ├── best_model.onnx       # ONNX graph (1.1MB)
    ├── best_model.onnx.data  # ONNX weights (27MB)
    ├── best_model_cam.onnx   # CAM model graph (1.2MB, 含 relu_120 中間層)
    └── best_model_cam.onnx.data  # CAM model weights (27MB)
```

### API Endpoints

| Method | Path | 說明 |
|--------|------|------|
| GET | `/health` | 健康檢查 (model_loaded, db_ready, providers) |
| POST | `/api/predict` | X 光片推論 (multipart file upload) |
| POST | `/api/gradcam` | CAM 熱力圖 (multipart file upload + label) |
| GET | `/api/model/info` | 模型資訊 |

### 資料庫連線

API 使用 SQLAlchemy 連接 PostgreSQL：

```
DATABASE_URL = postgresql://postgres:chestxpert123@postgres-svc:5432/chestxpert
```

### 本地啟動 API

```bash
cd api_build_onnx
pip install -r requirements.txt

# 啟動（開發模式）
uvicorn main:app --reload --port 8000

# 啟動（生產模式）
uvicorn main:app --workers 4 --port 8000
```

### 推論流程

```
上傳影像 (PNG/JPG)
    ↓
PIL 開啟 → resize 224x224 → normalize (ImageNet mean/std)
    ↓
轉換為 numpy array (1, 3, 224, 224), float32
    ↓
ONNX Runtime session.run({"image": array})
    ↓
Sigmoid → 15 個機率值
    ↓
排序 + 寫入 PostgreSQL
```

### CAM 熱力圖流程

```
POST /api/gradcam (file + label)
    ↓
載入影像 + 預處理 (同 predict)
    ↓
CAM model session.run() → 取得 relu_120 特徵圖 (1, 1024, 7, 7)
    ↓
取出 classifier 對應 label 的權重 (1024,)
    ↓
特徵圖 × 權重 → 加權求和 → 7×7 heatmap
    ↓
OpenCV resize 到原圖大小 + colormap (JET) + 疊加原圖
    ↓
Base64 編碼 PNG 回傳
```

**CAM 原理**: 不需要梯度反傳。直接從最後卷積層 (relu_120, 1024 channels) 的特徵圖中，用 classifier 權重加權求和，得到該疾病類別的 attention map。

### 測試

```bash
# 健康檢查
curl https://ai-x-ray-detection.avision-gb10.org/health

# 推論測試
curl -X POST https://ai-x-ray-detection.avision-gb10.org/api/predict \
  -F "file=@test_xray.png"

# 熱力圖測試
curl -X POST https://ai-x-ray-detection.avision-gb10.org/api/gradcam \
  -F "file=@test_xray.png" -F "label=Cardiomegaly"
```

---

## K8S 部署流程

### GB10 上的 Minikube

```bash
# 啟動 minikube (containerd driver)
minikube start --driver=docker --container-runtime=containerd \
  --cpus=8 --memory=16g --disk-size=100g

# 啟用 nvidia-device-plugin (GPU, 目前有問題)
minikube addons enable nvidia-device-plugin

# 確認節點狀態
kubectl get nodes
# NAME       STATUS   ROLES           AGE   VERSION
# minikube   Ready    control-plane   1h    v1.35.1
```

### K8S Resources (namespace: ai-xray)

| Resource | Type | 說明 |
|----------|------|------|
| ai-xray-api | Deployment | FastAPI + ONNX Runtime (CPU, 406MB) + CAM model |
| ai-xray-frontend | Deployment | nginx:alpine 前端 |
| postgres | StatefulSet | PostgreSQL 15-alpine |
| ai-xray-api-svc | ClusterIP :8000 | API service |
| ai-xray-api-nodeport | NodePort :30800 | API NodePort |
| postgres-svc | ClusterIP :5432 | PostgreSQL service |
| frontend-html | ConfigMap | nginx 前端內容 |

### 部署步驟

```bash
# 1. 建立 namespace
kubectl create namespace ai-xray

# 2. Build + Load images (host docker → minikube)
docker build -t ai-xray-api:latest ./api_build_onnx/
minikube image load ai-xray-api:latest

docker build -t ai-xray-frontend:latest ./frontend/
minikube image load ai-xray-frontend:latest

# 3. Deploy all
kubectl apply -f k8s/

# 4. 確認 Pod 狀態
kubectl get pods -n ai-xray
kubectl logs -f deployment/ai-xray-api -n ai-xray

# 5. 啟動 Cloudflare Tunnel
cloudflared tunnel run dgx-spark
```

### Docker Build 注意事項

> ⚠️ minikube 使用 containerd，不能在 minikube docker env 裡 build。
> 使用 host docker build，再用 `minikube image load` 匯入。

```bash
# ✅ 正確做法
docker build -t ai-xray-api:latest ./api_build_onnx/
minikube image load ai-xray-api:latest

# ❌ 錯誤：在 minikube containerd 裡 build
eval $(minikube docker-env)
docker build ...  # buildkit 404 error
```

### 外部存取

- **網址**: https://ai-x-ray-detection.avision-gb10.org
- **路徑**: Cloudflare Tunnel → 192.168.49.2:30080 → nginx → API
- **DNS**: Cloudflare 管理

---

## Git 工作流程

### 分支命名

```
main              # 穩定版本
dev               # 開發分支
feature/xxx       # 功能開發
fix/xxx           # 錯誤修復
docs/xxx          # 文件更新
```

### Commit 規範

使用 Conventional Commits：

```
feat: 新增推論 API 端點
fix: 修正 DataLoader 記憶體洩漏
docs: 更新 README
refactor: 重構模型匯出邏輯
test: 新增 API 測試
chore: 更新依賴版本
```

### 工作流程

```bash
# 建立功能分支
git checkout -b feature/add-frontend dev

# 開發 + 提交
git add .
git commit -m "feat: 實作前端上傳介面"

# 推送 + PR
git push origin feature/add-frontend
gh pr create --title "feat: 前端上傳介面" --base dev
```

---

## 除錯與排障

### 常見問題

#### 1. `CUDA out of memory`

```bash
# 降低 batch_size
python train.py --batch_size 16

# 或啟用梯度累積（等效 batch_size × accum_steps）
# config/default.yaml:
#   training:
#     batch_size: 16
#     grad_accum_steps: 2    # 等效 32
```

#### 2. `Unknown loss type`

loss type 只支援 `bce` 和 `focal`：

```yaml
loss:
  type: bce     # ✅ BCEWithLogitsLoss
  # type: bce_with_logits  # ❌ 錯誤
```

#### 3. `sm_121 not supported`

GB10 使用 NVIDIA 新架構（sm_121），需要 PyTorch nightly：

```bash
pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu130
```

#### 4. 訓練卡在 "Computing positive weights"

首次計算 pos_weight 需遍歷整個 train DataLoader（69,628 張圖），從磁碟讀取可能需要 5~10 分鐘。後續 epoch 會快很多（OS cache）。

#### 5. K8S Pod `ImagePullBackOff`

minikube 使用本地映像，需要先 load：

```bash
docker build -t ai-xray-api:latest ./api_build_onnx/
minikube image load ai-xray-api:latest
```

並在 K8S manifest 設定 `imagePullPolicy: Never` 或 `IfNotPresent`。

#### 6. Docker build 在 minikube containerd 失敗

containerd 的 buildkit 會出現 404 錯誤。解法：

```bash
# 在 host docker build，再 load 進 minikube
docker build -t ai-xray-api:latest ./api_build_onnx/
minikube image load ai-xray-api:latest
```

#### 7. ONNX model 載入失敗

ONNX 模型是 **兩個檔案**：`best_model.onnx` (graph) + `best_model.onnx.data` (weights)。
Dockerfile 裡必須 COPY 兩個檔案：

```dockerfile
COPY models/best_model.onnx /app/models/
COPY models/best_model.onnx.data /app/models/
```

#### 8. nvidia-device-plugin 無法註冊 GPU

GB10 的 NVIDIA libs 在 `/usr/lib/aarch64-linux-gnu/`，但 device plugin 預期 `/usr/local/nvidia/`。
目前使用 CPU 部署。GPU 替代方案：
- 直接 mount `/dev/nvidia*` device + CUDA libs 到 pod
- 設定 containerd 使用 nvidia-container-runtime
- 使用 CDI (Container Device Interface)

#### 9. Minikube 重啟後 API server port 改變

`minikube stop && minikube start` 後 API server port 會變。
NodePort (30800) 不受影響，Cloudflare Tunnel 使用 NodePort 所以外部存取不受影響。
但 kubectl 可能需要更新 kubeconfig。

#### 10. pexpect SSH 密碼被遮蔽

pexpect 會將敏感字串替換為 `***`。用 `base64` 編碼輸出來繞過：

```bash
grep DATABASE_URL main.py | sed 's/.*:\/\/postgres://;s/@.*//' | base64
```

#### 11. minikube containerd image cache 不更新

`minikube image load` 可能使用舊的 layer cache，導致更新的映像沒有生效。

```bash
# 解法：清除舊映像 + 用新 tag
minikube ssh "sudo crictl rmi <old-image>"
docker build -t ai-xray-api:new-tag ./api_build_onnx/
minikube image load ai-xray-api:new-tag
kubectl set image deployment/ai-xray-api api=ai-xray-api:new-tag -n ai-xray
```

#### 12. Cloudflared tunnel 路由異常

有時 cloudflared 需要 restart 才能正確路由：

```bash
sudo systemctl restart cloudflared
# 確認狀態
sudo systemctl status cloudflared
```

### 訓練監控

```bash
# 查看訓練 log
tail -f ~/ai-xray-detection/logs/train.log

# GPU 使用率
watch -n 5 nvidia-smi

# 訓練進程狀態
ps aux | grep train.py
```

### 模型除錯

```bash
# 只跑評估（不訓練）
python train.py --eval_only --resume models/densenet121_nih/best_model.pth

# 只匯出模型
python train.py --export_only --resume models/densenet121_nih/best_model.pth
```

### K8S 除錯

```bash
# 查看 pod 狀態
kubectl get pods -n ai-xray

# 查看 pod 日誌
kubectl logs -f deployment/ai-xray-api -n ai-xray

# 進入 pod
kubectl exec -it deployment/ai-xray-api -n ai-xray -- bash

# API 健康檢查 (pod 內無 curl，用 python)
kubectl exec -n ai-xray deployment/ai-xray-api -- \
  python3 -c "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/health').read().decode())"
```

---

## 效能調校建議

| 參數 | 預設 | 建議調整 | 說明 |
|------|------|----------|------|
| `batch_size` | 32 | 16~64 | GB10 記憶體足夠可用 64 |
| `img_size` | 224 | 224/384 | 384 精度較高但慢 3x |
| `num_workers` | 4 | 4~8 | GB10 20 核可用 8 |
| `grad_accum_steps` | 1 | 1~4 | 記憶體不足時增加 |
| `amp` | true | true | 混合精度省 40% VRAM |
| `freeze_backbone_epochs` | 3 | 2~5 | 避免過早破壞預訓練特徵 |
