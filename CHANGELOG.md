# 變更日誌

本專案所有值得注意的變更都會記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
並且遵守 [語意化版本](https://semver.org/lang/zh-TW/)。

---

## [0.1.0] - 2026-05-20

### 新增

#### 研究與規劃
- 完成胸腔 X 光影像分類研究報告：模型架構比較（DenseNet-121 / EfficientNet / ViT）、Transfer Learning 策略、訓練配置建議、GPU 需求評估
- 確立技術棧：PyTorch + DenseNet-121（Transfer Learning）+ FastAPI + PostgreSQL + Docker + Kubernetes
- 選定 NIH ChestX-ray14 資料集（108,948 張影像、14 種胸腔疾病 + No Finding 共 15 類標籤）

#### 資料管線（data\_etl）
- 完成 NIH ChestX-ray14 ETL Pipeline
  - 下載腳本（支援 Kaggle / TFDS / HuggingFace 三種來源）
  - CSV 元資料清洗：解析 `Data_Entry_2017_v2020.csv`、Multi-hot 編碼、Patient-level split（嚴格防止 train/val/test 洩漏）
  - 資料驗證腳本：洩漏檢查、分佈統計、DataLoader 測試
  - 一鍵 ETL 管線 `run_etl.sh`
- 實作 `ChestXray14Dataset` PyTorch Dataset / DataLoader
  - 灰階 → 3 通道（相容 ImageNet 預訓練 backbone）
  - 訓練資料增強：水平翻轉、仿射變換（±10° 旋轉）、色彩抖動
  - Median-freq 類別權重計算，處理嚴重類別不平衡（Hernia 0.2% vs Effusion 12.2%）
- 撰寫 31 個單元測試（mock data，不需真實資料集即可執行）

#### 資料庫層（database）
- 設計並實作五張核心表格：`users`、`patients`、`images`、`image_labels`、`predictions`
- ORM 模型（SQLAlchemy）：User、Patient、Image、ImageLabel、Prediction，含完整 constraint 與關聯
- Alembic migration 管理
  - 初始 schema migration（`001_initial.py`）
- CRUD 操作層
  - `CRUDBase` 泛型基底類別（支援 async）
  - 各 Entity 專屬 CRUD：User、Patient、Image、Prediction
- Pydantic schemas（API 請求/回應驗證）
  - User、Patient、Image、Prediction 的 create / update / response schema
- NIH CSV seed script（`seed_nih.py`）：批次匯入 108K 張影像元資料
- 環境變數管理（Pydantic Settings + `.env`）
- ERD 文件、索引策略、備份策略（每日 Full Backup + 每小時 WAL archiving）

#### 模型訓練（model\_training）
- 實作 `ChestXrayClassifier` 模型架構
  - 支援 backbone：DenseNet-121、EfficientNet-B0、EfficientNet-B3
  - ImageNet 預訓練權重 + 自訂多標籤分類頭（Dropout → Linear）
  - Backbone 凍結 / 解凍機制（Progressive Unfreezing）
  - 分層 Learning Rate（head: 1e-3, backbone: 1e-4）
- 完整訓練管線（`train.py`）
  - YAML 配置驅動（`config/default.yaml`）+ CLI 參數覆蓋
  - 混合精度訓練（AMP / GradScaler）
  - 梯度累積（Gradient Accumulation）
  - 梯度裁剪（Gradient Clipping）
  - 訓練恢復（Resume from checkpoint）
  - 支援純評估模式（`--eval_only`）與純匯出模式（`--export_only`）
- 損失函數
  - BCEWithLogitsLoss（含正樣本權重）
  - Focal Loss（alpha=0.25, gamma=2.0，處理類別不平衡）
- 評估指標模組
  - Per-class AUROC、Macro/Micro F1、Per-class PR-AUC
  - 最佳閾值搜尋（Precision-Recall curve 上 F1 最大化的 threshold）
  - 完整評估報告格式化輸出
- 學習率排程器
  - Cosine Annealing（預設）
  - ReduceLROnPlateau
  - StepLR
- 回調機制
  - Early Stopping（patience=7，monitor val\_auc）
  - Checkpoint Manager（best / last / top-k）
- 實驗追蹤
  - CSV logger（預設，無需外部依賴）
  - 相容 TensorBoard / W&B 介面
- 模型匯出
  - ONNX（opset 17，dynamic batch）
  - TorchScript
- NIH 資料集預處理腳本（`preprocess_nih.py`）

#### FastAPI 後端（api）
- 應用程式核心
  - FastAPI application factory + lifespan 事件管理
  - CORS 中介軟體
  - Rate Limiting（SlowAPI，預設 100 req/min）
- 認證系統
  - JWT（HS256）認證：註冊 / 登入 / 取得當前使用者
  - bcrypt 密碼雜湊
  - Token 過期管理（預設 60 分鐘）
- API 端點
  - `POST /auth/register` — 使用者註冊
  - `POST /auth/login` — 登入，取得 JWT token
  - `GET /auth/me` — 當前使用者資訊（需認證）
  - `POST /api/predict` — 上傳影像，取得多標籤分類結果
  - `GET /api/patients` — 病患列表（分頁）
  - `GET /api/patients/{id}` — 取得單一病患
  - `GET /api/patients/{id}/images` — 取得病患影像
  - `GET /api/predictions/{id}` — 取得預測詳情
  - `GET /api/model/info` — 模型版本與指標
  - `GET /health` — 健康檢查
- 推論服務（InferenceService）
  - 支援 ONNX Runtime 與 TorchScript 雙引擎
  - 單張與批次推論
  - 自動 sigmoid 後處理（multi-label）
  - 無模型時自動降級為 stub 模式
  - CPU / CUDA 裝置選擇
- 資料庫整合（SQLAlchemy async session）
- 影像預處理服務（正規化、縮放）
- Pydantic Settings 配置管理
- Dockerfile（多階段建構）
- 測試
  - API 端點測試（`test_api.py`）
  - 認證測試（`test_auth.py`）
  - 服務層測試（`test_services.py`）

#### 基礎設施（k8s）
- Kubernetes 部署配置
  - Namespace：`ai-xray`
  - Deployment：FastAPI 後端（含 GPU 資源請求、PVC 掛載）
  - Service：ClusterIP 暴露 HTTP 8000

### 待開發

- 前端網頁介面（React）
- 完整 CI/CD Pipeline
- 模型實際訓練結果與調參紀錄
- 生產環境部署配置（Ingress、HPA、Monitoring）

---

[0.1.0]: https://github.com/user/AI-X-ray-detection/releases/tag/v0.1.0
