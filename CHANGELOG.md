# 變更日誌

本專案所有值得注意的變更都會記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
並且遵守 [語意化版本](https://semver.org/lang/zh-TW/)。

---

## [0.6.0] - 2026-05-21

### Added
- **API Key 認證 + Rate Limiting**: Header-based `X-API-Key`，env `API_KEYS` 設定
  - Opt-in 設計：未設定 `API_KEYS` 環境變數 = 認證關閉，完全向後相容
  - `/health` endpoint 永遠公開，不需 API Key
  - 速率限制：100 req/min per key（env 可調）
- **CORS 白名單**: env `CORS_ORIGINS` 設定允許的域名（逗號分隔），預設 `["*"]`
- **36 單元測試**: pytest 完整覆蓋（predict, gradcam, DICOM, auth, rate limiting, edge cases）
- **CI/CD Pipeline**: GitHub Actions — lint (ruff) + test (3.10/3.11/3.12) + Docker build

### Changed
- **Dev URL 更名**: `ai-xray-test.avision-gb10.org` → `dev-ai-x-ray-detection.avision-gb10.org`
  - 一層子域名相容 Cloudflare Universal SSL
  - Cloudflare tunnel 新增 CNAME `dev-ai-x-ray-detection` → `192.168.49.2:30081`
- **UI 判讀流程優化**: 判讀後自動顯示左原圖 + 右熱力圖（自動呼叫 Grad-CAM API）
- **PDF 報告重寫**: Canvas 渲染中文（NotoSansTC/PingFang TC），影像按原比例顯示
  - 報告含：標題 banner + 診斷卡片 + 原圖/熱力圖並排 + 14 種疾病彩色條狀圖 + 免責聲明
- **移除按鈕修正**: 按「移除」後按鈕文字還原為「開始判讀」
- K8S `deployment.yaml` 新增 `API_KEYS` + `CORS_ORIGINS` 環境變數

---

## [0.5.0] - 2026-05-21

### Added
- **DICOM (.dcm) 格式支援**: pydicom 整合，自動偵測 DICOM 檔案
  - `is_dicom()`: 檢查 DICM magic bytes 自動偵測 DICOM 格式
  - `dicom_to_image()`: 灰階正規化 + 多通道處理，轉換為模型可用的 RGB 影像
  - 前端: `.dcm` 格式 accept + DICOM 預覽 SVG placeholder
  - `/api/predict` 和 `/api/gradcam` 均支援 DICOM 輸入
- **手機版 RWD**: 24 條 responsive CSS rules，`@media (max-width: 640px)` 適配手機螢幕
- **PDF 報告匯出**: jsPDF CDN 整合，一鍵產生報告
  - 報告內容: 標題 + 最高信心診斷 + 15 種疾病信心度條狀圖 + 免責聲明
  - 前端結果頁新增 PDF 匯出按鈕
- **批次上傳**: 多檔案選擇器，循序 API 呼叫，即時進度顯示
- **docker-compose.yml**: 本地開發一鍵啟動
  - API (`:8000`) + Frontend (`:3000`) + PostgreSQL (`:5432`)
  - Health checks、named volume (`pgdata`)
- **start.bat**: Windows 一鍵啟動腳本
  - 自動建立 venv → pip install → 啟動 uvicorn (`:8000`) + http.server (`:3000`)
  - 不需要 Docker 或 PostgreSQL
- **測試環境**: `dev-ai-x-ray-detection.avision-gb10.org` (NodePort 30081)
  - 獨立 K8S deployments: `ai-xray-api-test`, `ai-xray-frontend-test`
  - Cloudflare tunnel: `dev-ai-x-ray-detection.avision-gb10.org` → `192.168.49.2:30081`

### Changed
- **前端 API_BASE 自動偵測**: Frontend 在 `:3000` 執行時自動指向 API `:8000`，無需手動設定
- **Input validation 加強**:
  - 檔案大小上限: 50MB
  - DICOM 副檔名檢查 (`.dcm`)
- `api_build_onnx/requirements.txt` 新增 `pydicom>=2.4.0`

---

## [0.4.0] - 2026-05-21

### Added
- **CAM 熱力圖視覺化**: 上傳 X 光片後可查看模型關注區域
  - API: `POST /api/gradcam` — 回傳 base64 heatmap + 預測結果
  - 前端: 🔥 熱力圖按鈕 → 並排顯示原圖/疊加圖，可下拉切換 14 種疾病
  - 方法: CAM (Class Activation Mapping) — 最後卷積層 `relu_120` 特徵圖 × classifier 權重
  - ONNX CAM model: `best_model_cam.onnx` (含 relu_120 中間層輸出, 1024 channels, 7×7)
  - 推論速度: ~345ms (含 heatmap 生成)
- **前端 UI 更新**: 疾病選擇下拉選單、熱力圖疊加顯示、深色主題一致

### Changed
- `api_build_onnx/requirements.txt` 新增 `onnx>=1.16.0`, `opencv-python-headless>=4.8.0`
- `api_build_onnx/Dockerfile` 新增 `libgl1 libglib2.0-0` (OpenCV deps)
- 前端 nginx root 改為 `/tmp` (workaround for read-only FS in minikube containerd)

### Fixed
- ONNX model data 損壞問題 — `best_model.onnx.data` MD5 不符導致 ORT 1.26.0 載入失敗
- minikube containerd image cache — 需 `crictl rmi` + 新 tag 才能更新映像
- Cloudflared tunnel 重啟後恢復正常路由

---

## [0.3.0] - 2026-05-21

### Changed
- **API 容器映像大幅縮小**: 從 5.15GB (PyTorch) → 406MB (ONNX Runtime)
- 建置路徑從 `api_build/` 遷移至 `api_build_onnx/`
- minikube driver 從 docker 改為 containerd
- K8S 叢集重建 (minikube v1.35.1, containerd)

### Added
- **PostgreSQL 整合**: 預測結果自動寫入資料庫
  - StatefulSet `postgres-0` (PostgreSQL 15-alpine)
  - predictions 表: id, filename, content_type, file_size_kb, top_prediction, top_confidence, all_results (JSONB), processing_time_ms, created_at
- **ONNX 模型推論**: 從 PyTorch 改為 ONNX Runtime
  - 模型: `best_model.onnx` (1.1MB) + `best_model.onnx.data` (27MB)
  - Provider: CPUExecutionProvider
- **API health endpoint** 新增 `db_ready` 和 `providers` 欄位
- **ONNX Runtime GPU wheel**: 從 source build 成功 (v1.24.4, aarch64 + CUDA, 54MB)
- **GPU Docker image**: `ai-xray-api:gpu` (598MB) 已建置 (尚未部署)
- NodePort Service: `ai-xray-api-nodeport` (30800)

### Performance
- 推論速度: ~90ms/張 (CPU, ARM ONNX Runtime) ← 從 ~1.5s (PyTorch) 改善

### Infrastructure
- Docker build 改為 host docker + `minikube image load` (containerd 不支援直接 build)
- PostgreSQL credentials: postgres/chestxpert123@postgres-svc/chestxpert
- ConfigMap `frontend-html` for nginx content

### Known Issues
- nvidia-device-plugin v0.18.2/v0.14.5 無法在 minikube containerd 註冊 GPU resource
  - 原因: `/usr/local/nvidia/` 不存在 (GB10 libs 在 `/usr/lib/aarch64-linux-gnu/`)
  - GPU image 已建好但暫時使用 CPU 部署

---

## [0.2.0] - 2026-05-20

### Added
- 前端網頁：深色主題 UI，支援拖曳上傳 X 光片
- 前端顯示 14 種疾病 + No Finding 的機率條狀圖（中英雙語）
- FastAPI 推論服務容器化部署 (ai-xray-api)
- Nginx 前端容器化部署 (ai-xray-frontend)，含 API reverse proxy
- K8S 部署：frontend (NodePort 30800) + api (ClusterIP 8000)
- Cloudflare Tunnel 外部存取：https://ai-x-ray-detection.avision-gb10.org
- API endpoint：POST /api/predict, GET /health, GET /api/model/info
- 前端 /health proxy 到後端

### Infrastructure
- Minikube 叢集 (Docker driver, Calico CNI)
- Docker images: ai-xray-frontend (61.5MB nginx:alpine), ai-xray-api (5.15GB pytorch)
- K8S namespace: ai-xray

---

## [0.1.0] - 2026-05-20

### Added
- DenseNet-121 模型訓練 (NIH ChestX-ray14, Mean AUROC 0.812)
- 資料預處理腳本 (preprocess_nih.py)
- 模型匯出 (best_model.pth + best_model.onnx)
- 專案初始化 (git repo, 資料集, 訓練日誌)
