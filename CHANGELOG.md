# 變更日誌

本專案所有值得注意的變更都會記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.1.0/)，
並且遵守 [語意化版本](https://semver.org/lang/zh-TW/)。

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
