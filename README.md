# 🫁 AI X-ray Detection — 胸腔 X 光片自動判讀系統

> 基於深度學習的胸腔 X 光片多標籤分類系統  
> 使用 NIH ChestX-ray14 資料集（108,948 張影像、14 種胸腔疾病標籤）  
> 訓練 DenseNet-121 模型，並以 FastAPI + Kubernetes 部署

---

## 📖 專案簡介

本系統實作了從資料前處理、模型訓練到 API 推論部署的完整端到端管線。使用者上傳胸腔 X 光片後，系統自動辨識 **14 種常見胸腔疾病**，回傳各疾病的信心分數與建議，供臨床人員參考。

**14 種可辨識疾病標籤：**

```
Atelectasis（肺不張）         Cardiomegaly（心臟肥大）
Consolidation（肺實變）        Edema（肺水腫）
Effusion（胸腔積液）           Emphysema（肺氣腫）
Fibrosis（肺纖維化）          Hernia（橫膈膜疝氣）
Infiltration（肺浸潤）         Mass（腫塊）
Nodule（肺結節）              Pleural_Thickening（肋膜增厚）
Pneumonia（肺炎）             Pneumothorax（氣胸）
```

> ⚠️ **免責聲明**：本系統僅供學術研究與技術展示用途，不得作為臨床診斷依據。

---

## 🏗️ 系統架構

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          系統架構總覽                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────┐    HTTPS     ┌─────────────────────────────────────┐     │
│  │  使用者   │ ──────────> │        FastAPI 後端 (:8000)          │     │
│  │ (前端/APP)│ <────────── │                                     │     │
│  └──────────┘              │  ┌───────────┐  ┌────────────────┐  │     │
│                            │  │ /auth/*   │  │  /api/*        │  │     │
│                            │  │ JWT 認證  │  │  核心業務 API  │  │     │
│                            │  └───────────┘  └───────┬────────┘  │     │
│                            │                          │           │     │
│                            │  ┌───────────────────────┴───────┐  │     │
│                            │  │       推論服務層               │  │     │
│                            │  │  Preprocess → ONNX/TorchScript │  │     │
│                            │  │  (sigmoid → 14 類信心分數)     │  │     │
│                            │  └───────────────────────────────┘  │     │
│                            └──────────────┬──────────────────────┘     │
│                                           │                             │
│                            ┌──────────────┴──────────────┐             │
│                            │       PostgreSQL 資料庫      │             │
│                            │  users / patients / images  │             │
│                            │  image_labels / predictions │             │
│                            └─────────────────────────────┘             │
│                                                                         │
│  ┌──────────────────┐  ┌───────────────────┐  ┌──────────────────┐    │
│  │  Data ETL 管線   │  │  模型訓練管線      │  │  K8s 部署        │    │
│  │  (data_etl/)     │  │  (model_training/) │  │  (k8s/)          │    │
│  │                  │  │                    │  │                  │    │
│  │  NIH CSV 下載    │  │  DenseNet-121      │  │  Deployment      │    │
│  │  清洗 → 分割    │  │  Transfer Learning │  │  Service         │    │
│  │  train/val/test  │  │  ONNX / TorchScript│  │  PVC 掛載        │    │
│  └──────────────────┘  └───────────────────┘  └──────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ✨ 功能特色

- **多標籤分類**：同時預測 14 種胸腔疾病，支援一張影像多種病徵
- **多骨幹架構**：支援 DenseNet-121、EfficientNet-B0/B3 切換
- **Transfer Learning**：ImageNet 預訓練 + 漸進式解凍策略
- **類別不平衡處理**：加權 BCE Loss（pos_weight）與 Focal Loss
- **進階訓練技巧**：AMP 混合精度、梯度累積、梯度裁剪、Cosine Scheduler
- **自動閾值調優**：驗證集上自動尋找各類別 F1 最優閾值
- **完整評估指標**：AUROC、PR-AUC、F1（Macro/Micro/Weighted/Sample）
- **Early Stopping + Checkpoint**：patience=7 監控 val_auc，保留 top-3 最佳模型
- **ONNX / TorchScript 匯出**：支援生產環境高效推論
- **RESTful API**：JWT 認證、Rate Limiting、分頁查詢、Swagger 文件
- **DICOM 支援**：接受 PNG、JPEG、DICOM 格式上傳
- **Docker + Kubernetes**：容器化部署、GPU 資源調度、健康檢查
- **Patient-level Split**：依病患切割訓練/驗證集，避免資料洩漏
- **CLAHE 對比度增強**：可選的影像前處理技術

---

## 🛠️ 技術棧

**機器學習**
- PyTorch 2.2+ / torchvision — 模型訓練與推論
- ONNX Runtime 1.18+ — 高效推論引擎
- scikit-learn — 評估指標計算
- OpenCV (CLAHE) — 影像前處理
- numpy / pandas — 資料處理

**後端 API**
- FastAPI 0.115 — 非同步 Web 框架
- SQLAlchemy 2.0 + Alembic — ORM + 資料庫遷移
- PostgreSQL — 關聯式資料庫
- python-jose + passlib — JWT 認證 + bcrypt 密碼雜湊
- slowapi — API 速率限制

**基礎設施**
- Docker (python:3.11-slim) — 容器化
- Kubernetes — 叢集編排與 GPU 調度
- uvicorn — ASGI 伺服器

---

## 📁 目錄結構

```
AI-X-ray-detection/
├── README.md                          # 專案主文件（本檔案）
├── docs/
│   └── research/
│       └── chest-xray-training-guide.md  # 訓練指南文件
│
├── data_etl/                          # 資料前處理管線
│   ├── src/
│   │   ├── __init__.py
│   │   └── chestxray14_dataset.py     # NIH 資料集 Dataset/DataLoader
│   └── tests/
│       └── test_etl.py
│
├── database/                          # 資料庫層
│   ├── alembic/
│   │   ├── versions/
│   │   │   └── 001_initial.py         # 初始 schema 遷移
│   │   └── script.py.mako
│   ├── scripts/
│   │   └── seed_nih.py                # NIH CSV → PostgreSQL 匯入腳本
│   └── requirements.txt
│
├── model_training/                    # 模型訓練管線
│   ├── config/
│   │   └── default.yaml               # 訓練超參數設定
│   ├── src/
│   │   ├── model.py                   # ChestXrayClassifier (DenseNet/EfficientNet)
│   │   ├── dataset.py                 # 訓練用 Dataset (CheXpert 格式)
│   │   ├── losses.py                  # BCEWithLogitsLoss / FocalLoss
│   │   ├── metrics.py                 # AUROC / PR-AUC / F1 / 閾值調優
│   │   ├── callbacks.py               # EarlyStopping / Checkpoint / Scheduler
│   │   └── tracker.py                 # 實驗追蹤 (CSV logging)
│   ├── train.py                       # 主訓練腳本
│   ├── preprocess_nih.py              # NIH CSV 前處理 → train/val/test CSV
│   └── requirements.txt
│
├── api/                               # FastAPI 後端服務
│   ├── src/
│   │   ├── main.py                    # App factory + 生命週期管理
│   │   ├── config.py                  # Pydantic Settings (環境變數)
│   │   ├── database.py                # SQLAlchemy 連線設定
│   │   ├── models.py                  # ORM 模型 (User/Patient/Image/Prediction)
│   │   ├── schemas.py                 # Pydantic 請求/回應 schema
│   │   ├── crud.py                    # 資料庫 CRUD 操作
│   │   ├── api/
│   │   │   ├── auth.py                # /auth/* 認證端點
│   │   │   └── core.py                # /api/* 核心業務端點
│   │   ├── services/
│   │   │   ├── auth.py                # JWT 服務 + 密碼驗證
│   │   │   ├── inference.py           # ONNX/TorchScript 推論服務
│   │   │   └── preprocessing.py       # 影像前處理管線
│   │   └── middleware/
│   │       └── rate_limit.py          # API 速率限制
│   ├── tests/
│   │   ├── test_api.py
│   │   ├── test_auth.py
│   │   ├── test_services.py
│   │   └── conftest.py
│   ├── Dockerfile
│   ├── requirements.txt
│   └── pytest.ini
│
├── k8s/                               # Kubernetes 部署配置
│   ├── namespace.yaml
│   ├── deployment.yaml                # API Deployment (GPU + PVC)
│   └── service.yaml
│
└── frontend/                          # 前端網頁（規劃中）
```

---

## 📋 環境需求

**系統**
- Python 3.11+
- PostgreSQL 14+
- CUDA 11.8+（GPU 訓練/推論選配）

**訓練 GPU 建議**
- 最低：NVIDIA RTX 3060 (12GB VRAM) — 需梯度累積
- 建議：NVIDIA RTX 4090 (24GB VRAM) — batch_size=32 穩定訓練
- 雲端：NVIDIA A100 (40GB) — 全資料集完整訓練

---

## 🚀 快速開始

### 1. 下載與前處理資料集

```bash
# 下載 NIH ChestX-ray14 資料集
# https://nihcc.app.box.com/s/vfk49d74nhbxq3nqjxj9xj5tz2mncv5p

# 前處理：產生 train.csv / val.csv / test.csv
cd model_training
python preprocess_nih.py \
  --data_dir /path/to/nih-chest-xrays \
  --output_dir /path/to/nih-chest-xrays
```

### 2. 初始化資料庫

```bash
cd database

# 安裝依賴
pip install -r requirements.txt

# 執行遷移
alembic upgrade head

# 匯入 NIH 資料（選配）
python scripts/seed_nih.py \
  --csv /path/to/Data_Entry_2017_v2020.csv \
  --images-dir /path/to/nih-chest-xrays/images
```

### 3. 訓練模型

```bash
cd model_training

# 安裝依賴
pip install -r requirements.txt

# 使用預設設定訓練（DenseNet-121, 30 epochs）
python train.py --config config/default.yaml

# 自訂參數
python train.py \
  --config config/default.yaml \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.0005 \
  --device cuda:0

# 僅評估（不訓練）
python train.py --config config/default.yaml --eval_only --resume models/best_model.pth

# 僅匯出 ONNX/TorchScript
python train.py --config config/default.yaml --export_only --resume models/best_model.pth
```

訓練輸出檔案：
- `best_model.pth` — 最佳模型檢查點
- `model.onnx` — ONNX 匯出
- `model.torchscript` — TorchScript 匯出
- `training_results.json` — 訓練歷史與指標
- `evaluation_report.txt` — 評估報告

### 4. 啟動 API 服務

```bash
cd api

# 安裝依賴
pip install -r requirements.txt

# 設定環境變數
cp .env.example .env
# 編輯 .env 設定 DATABASE_URL, JWT_SECRET_KEY, MODEL_PATH 等

# 啟動開發伺服器
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 開啟互動式 API 文件
# http://localhost:8000/docs        (Swagger UI)
# http://localhost:8000/redoc       (ReDoc)
```

### 5. Docker 部署

```bash
# 建置映像
docker build -t ai-xray-api:latest -f api/Dockerfile api/

# 執行容器
docker run -d \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/chestxray \
  -e MODEL_PATH=/app/models/model.onnx \
  -v /path/to/models:/app/models \
  ai-xray-api:latest
```

### 6. Kubernetes 部署

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

---

## 🔌 API 端點

### 認證 (`/auth`)

| 方法 | 端點 | 說明 | 認證 |
|------|------|------|------|
| `POST` | `/auth/register` | 註冊新使用者 | ❌ |
| `POST` | `/auth/login` | 登入取得 JWT Token | ❌ |
| `GET` | `/auth/me` | 取得當前使用者資訊 | ✅ |

### 核心業務 (`/api`)

| 方法 | 端點 | 說明 | 認證 |
|------|------|------|------|
| `POST` | `/api/predict` | 上傳胸腔 X 光片進行判讀 | 選配 |
| `GET` | `/api/patients` | 列出病患（分頁） | ❌ |
| `GET` | `/api/patients/{id}` | 取得單一病患資訊 | ❌ |
| `GET` | `/api/patients/{id}/images` | 取得病患的影像紀錄 | ❌ |
| `GET` | `/api/predictions/{id}` | 取得預測結果詳細資訊 | ❌ |
| `GET` | `/api/model/info` | 取得模型版本與資訊 | ❌ |

### 系統

| 方法 | 端點 | 說明 |
|------|------|------|
| `GET` | `/health` | 健康檢查 |
| `GET` | `/docs` | Swagger UI 互動式文件 |
| `GET` | `/redoc` | ReDoc 文件 |

### 範例：上傳影像進行判讀

```bash
# 上傳 X 光片（匿名）
curl -X POST http://localhost:8000/api/predict \
  -F "file=@chest_xray.png"

# 上傳 X 光片（附帶認證）
curl -X POST http://localhost:8000/api/predict \
  -H "Authorization: Bearer <your-jwt-token>" \
  -F "file=@chest_xray.png"
```

**回傳範例：**
```json
{
  "id": "a1b2c3d4-...",
  "model_version": "1.0.0",
  "results": [
    {"label": "Atelectasis", "confidence": 0.892},
    {"label": "Effusion", "confidence": 0.734},
    {"label": "Cardiomegaly", "confidence": 0.521}
  ],
  "top_prediction": "Atelectasis",
  "top_confidence": 0.892,
  "processing_time_ms": 45.32,
  "created_at": "2026-05-20T12:00:00Z"
}
```

---

## ⚙️ 環境變數

在 `api/` 目錄下建立 `.env` 檔案進行設定：

| 變數名稱 | 預設值 | 說明 |
|---------|--------|------|
| `APP_NAME` | `ChestXray API` | 應用程式名稱 |
| `APP_VERSION` | `1.0.0` | 版本號 |
| `DEBUG` | `True` | 除錯模式（生產環境請設為 False） |
| `DATABASE_URL` | `postgresql://postgres:***@localhost:5432/chestxray` | PostgreSQL 連線字串 |
| `JWT_SECRET_KEY` | `change-me-in-production...` | JWT 簽章金鑰（**生產環境必須更換**） |
| `JWT_ALGORITHM` | `HS256` | JWT 演算法 |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `60` | Token 有效期限（分鐘） |
| `RATE_LIMIT` | `100/minute` | API 速率限制 |
| `RATE_LIMIT_ENABLED` | `True` | 是否啟用速率限制 |
| `MODEL_PATH` | `models/model.onnx` | 模型檔案路徑（.onnx / .pt / .pth） |
| `MODEL_DEVICE` | `cpu` | 推論裝置（`cpu` / `cuda` / `cuda:0`） |
| `MODEL_BATCH_SIZE` | `8` | 批次推論大小 |
| `CONFIDENCE_THRESHOLD` | `0.5` | 信心分數顯示閾值 |

---

## 🗺️ Roadmap

- [x] NIH ChestX-ray14 資料前處理管線
- [x] DenseNet-121 多標籤分類模型訓練
- [x] ONNX / TorchScript 模型匯出
- [x] FastAPI 推論 API（JWT 認證、Rate Limiting）
- [x] PostgreSQL 資料庫 schema + Alembic 遷移
- [x] Docker 容器化 + Kubernetes 部署配置
- [ ] 前端網頁介面（React）
- [ ] EfficientNet-B3 / ViT 模型實驗
- [ ] DICOM 格式完整支援（pydicom 整合）
- [ ] 模型效能基準測試與結果發布
- [ ] GradCAM 視覺化（熱力圖顯示模型關注區域）
- [ ] 批次推論 API（多張影像同時上傳）
- [ ] CI/CD Pipeline（GitHub Actions）
- [ ] 模型監控與效能追蹤（Prometheus + Grafana）
- [ ] 多語系支援

---

## 👥 團隊

四人小組，三個月開發週期。大三專案。

---

## 📄 License

MIT License

---

## 📚 參考文獻

- Wang X, Peng Y, Lu L, et al. *ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases.* CVPR 2017.
- Rajpurkar P, Irvin J, Zhu K, et al. *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv 2017.
- Irvin J, Rajpurkar P, Ko M, et al. *CheXpert: A Large Chest Radiograph Dataset for Uncertainty and Comparison.* AAAI 2019.
- Huang G, Liu Z, van der Maaten L, Weinberger KQ. *Densely Connected Convolutional Networks.* CVPR 2017.
