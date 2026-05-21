# 🫁 AI X-ray Detection — 胸腔 X 光片自動判讀系統

大三專題：基於深度學習的胸腔 X 光片多標籤疾病分類系統。

## 架構

```
瀏覽器 → https://ai-x-ray-detection.avision-gb10.org
                    │
                    ▼
          ┌─ Nginx Frontend (NodePort 30080)
          │   ├─ /          → 靜態網頁 (上傳 + 結果顯示)
          │   └─ /api/*     → reverse proxy → API Service
          │
          ├─ FastAPI Backend (ai-xray-api-svc:8000)
          │   ├─ POST /api/predict   → X 光片推論
          │   ├─ POST /api/gradcam   → 熱力圖視覺化
          │   ├─ GET  /api/model/info → 模型資訊
          │   └─ GET  /health         → 健康檢查
          │
          └─ PostgreSQL (postgres-svc:5432)
              └─ predictions table (預測歷史記錄)
```

部署在 NVIDIA DGX Spark (GB10) 上的 Minikube K8S 叢集。

## 線上 Demo

👉 **https://ai-x-ray-detection.avision-gb10.org**

直接上傳胸腔 X 光片，即可獲得 14 種疾病的預測結果。也可使用 `data/test_images/` 中的 100 張範例影像進行測試。

```bash
# 或用 curl 測試
curl -X POST https://ai-x-ray-detection.avision-gb10.org/api/predict \
  -F "file=@data/test_images/test_009_Cardiomegaly.png"
```

## 模型

- **架構**: DenseNet-121 ( pretrained ImageNet → fine-tuned )
- **推論引擎**: ONNX Runtime (CPUExecutionProvider)
- **熱力圖**: CAM (Class Activation Mapping) — 最後卷積層 relu_120 (1024 channels) × classifier 權重
- **資料集**: NIH ChestX-ray14 ( 112,120 張正面胸腔 X 光片 )
- **標籤數**: 15 ( 14 種疾病 + No Finding )
- **輸入**: 224×224 RGB
- **輸出**: 15 個 sigmoid 機率值 ( 多標籤分類 )
- **Mean AUROC**: 0.812
- **推論速度**: ~90ms/張 (CPU, ARM ONNX Runtime)

### 支援的 14 種疾病

Atelectasis（肺不張）、Cardiomegaly（心臟肥大）、Consolidation（肺實變）、
Edema（肺水腫）、Effusion（胸腔積液）、Emphysema（肺氣腫）、
Fibrosis（肺纖維化）、Hernia（橫膈膜疝氣）、Infiltration（肺浸潤）、
Mass（腫塊）、Nodule（肺結節）、Pleural Thickening（肋膜增厚）、
Pneumonia（肺炎）、Pneumothorax（氣胸）

## 專案結構

```
ai-xray-detection/
├── frontend/              # 前端 (HTML/CSS/JS + Nginx)
│   ├── index.html
│   ├── nginx.conf
│   └── Dockerfile
├── api_build_onnx/        # API (FastAPI + ONNX Runtime) ← 使用中
│   ├── main.py            # FastAPI 推論服務 + CAM 熱力圖
│   ├── models/            # ONNX 模型 (best_model.onnx + .data + best_model_cam.onnx + .data)
│   ├── requirements.txt
│   ├── Dockerfile         # CPU 版 (406MB)
│   └── Dockerfile.gpu     # GPU 版 (598MB, 未部署)
├── api_build/             # 舊版 API (PyTorch, 5.15GB) ← 已停用
├── model_training/        # 訓練程式碼
│   ├── train.py
│   ├── preprocess_nih.py
│   └── requirements.txt
├── data/                  # 資料集 CSV + 影像
├── models/                # 訓練產出 (checkpoints, ONNX)
├── k8s/                   # K8S manifests
├── database/              # DB schema (Alembic)
├── api/                   # 完整 API (含 DB, 認證等)
├── CHANGELOG.md
├── DEVELOPMENT.md
└── TODO.md
```

## 資料集

### NIH ChestX-ray14

本專案使用 **NIH ChestX-ray14** 資料集進行訓練，包含 112,120 張正面胸腔 X 光片，涵蓋 14 種疾病標籤。

**下載位置（擇一）：**

| 來源 | 連結 | 說明 |
|------|------|------|
| Kaggle | https://www.kaggle.com/datasets/nih-chest-xrays/data | `kaggle datasets download nih-chest-xrays/data`（最方便） |
| HuggingFace | https://huggingface.co/datasets/BahaaEldin0/NIH-Chest-Xray-14 | Parquet 格式，`datasets` 庫可直接載入 |
| NIH 官方 | https://nihcc.app.box.com/s/vfk49d74nhbxq3nqjxj9 | 原始出處（Box 空間可能已失效，建議用上方鏡像） |

**下載後的目錄結構：**

```
data/
├── Data_Entry_2017.csv          # 標籤 metadata (從 Box 下載)
├── images_001/.../images/       # 影像分卷 (images_001 ~ images_012)
│   ├── 00000001_000.png
│   ├── 00000001_001.png
│   └── ...
├── train_list.txt               # 官方 train/val/test split
├── val_list.txt
└── test_list.txt
```

**前置處理（生成 train.csv / val.csv）：**

```bash
cd model_training
pip install -r requirements.txt
python preprocess_nih.py --data_dir /path/to/data --output_dir /path/to/data
```

### 預訓練 ONNX 模型

不想重新訓練的話，可直接使用 repo 中的 ONNX 模型進行推論：

```
api_build_onnx/models/
├── best_model.onnx       (1.1 MB — ONNX graph)
├── best_model.onnx.data  (27 MB  — ONNX weights)
├── best_model_cam.onnx   (1.2 MB — CAM model graph, 含 relu_120 中間層輸出)
└── best_model_cam.onnx.data (27 MB — CAM model weights)
```

> 模型已包含在 repo 中，clone 後即可直接 build Docker image。

## 快速開始

### 前端 (Nginx)

```bash
cd frontend
docker build -t ai-xray-frontend:latest .
minikube image load ai-xray-frontend:latest
kubectl apply -f k8s-frontend.yaml
```

### API (FastAPI + ONNX)

```bash
cd api_build_onnx
docker build -t ai-xray-api:latest .
minikube image load ai-xray-api:latest
```

> ⚠️ minikube 使用 containerd，不能在 minikube docker env 裡直接 build。
> 請在 host docker build 後用 `minikube image load` 匯入。

### 測試

```bash
# 健康檢查
curl https://ai-x-ray-detection.avision-gb10.org/health
# → {"status":"ok","model_loaded":true,"db_ready":true,"providers":["CPUExecutionProvider"]}

# 推論
curl -X POST https://ai-x-ray-detection.avision-gb10.org/api/predict \
  -F "file=@chest_xray.png"
# → {"results":[...],"top_prediction":"Cardiomegaly","top_confidence":0.9996,"processing_time_ms":90.29}

# 熱力圖 (CAM)
curl -X POST https://ai-x-ray-detection.avision-gb10.org/api/gradcam \
  -F "file=@chest_xray.png" \
  -F "label=Cardiomegaly"
# → {"heatmap":"<base64 PNG>","label":"Cardiomegaly","confidence":0.9996,...}
```

## 環境

- **硬體**: NVIDIA DGX Spark / GB10 (aarch64, 20-core ARM, Grace Hopper GPU, CUDA 13.2)
- **K8S**: Minikube v1.35.1 (containerd driver, K8S v1.35.1)
- **推論**: ONNX Runtime 1.24.4 (CPU, 已建 GPU wheel 待部署)
- **資料庫**: PostgreSQL 15-alpine (user: postgres, db: chestxpert)
- **外部存取**: Cloudflare Tunnel → ai-x-ray-detection.avision-gb10.org

## License

學術專題作品，僅供研究與展示用途。
