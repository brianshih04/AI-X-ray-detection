# TODO

## 已完成 ✅

- [x] 資料集準備 (NIH ChestX-ray14)
- [x] 資料預處理 (CSV 轉換, train/val/test split)
- [x] DenseNet-121 模型訓練 (Mean AUROC 0.812)
- [x] 模型評估 + ONNX 匯出
- [x] FastAPI 推論 API (ONNX Runtime, POST /api/predict)
- [x] 前端 UI (深色主題, 拖曳上傳, 14 疾病條狀圖)
- [x] 前端歷史記錄頁面 (GET /api/predictions, 表格呈現)
- [x] Docker 容器化 (frontend + API 406MB)
- [x] K8S 部署 (Minikube v1.35.1, containerd, 5 manifests)
- [x] K8S Secret 管理 (DB 密碼不寫死在程式碼)
- [x] Cloudflare Tunnel 外部存取 (frontend nginx → API proxy)
- [x] PostgreSQL 整合 (預測結果儲存)
- [x] E2E 測試通過 (CPU, ~90ms/張)
- [x] ONNX 模型 commit 到 GitHub (28MB, 免重新訓練)
- [x] 100 張測試影像 + NIH ChestX-ray14 下載指南
- [x] ONNX Runtime GPU wheel 建置 (aarch64 + sm_121, 54MB)
- [x] GPU Docker image 建置 (598MB)
- [x] CAM 熱力圖視覺化 (Class Activation Mapping)
  - API: POST /api/gradcam (base64 heatmap + 預測結果)
  - 前端: 🔥 按鈕 → 並排原圖/疊加圖，疾病下拉選單
  - CAM model: best_model_cam.onnx (relu_120 中間層, 1024×7×7)
  - ~345ms (含 heatmap 生成)
- [x] 文件更新 (README, CHANGELOG, DEVELOPMENT, TODO)
- [x] DICOM (.dcm) 格式支援
  - pydicom 整合，is_dicom() 自動偵測 (DICM magic)
  - dicom_to_image() 灰階正規化 + 多通道處理
  - 前端 .dcm accept + SVG 預覽 placeholder
  - /api/predict 和 /api/gradcam 均支援 DICOM
- [x] 手機版 RWD (24 條 responsive CSS rules, max-width: 640px)
- [x] 結果匯出 (PDF 報告)
  - Canvas 中文渲染 (NotoSansTC/PingFang TC)，含原圖 + 熱力圖 + 彩色條狀圖 + 免責聲明
- [x] 支援批次上傳 (多檔案選擇器，循序 API 呼叫，即時進度)
- [x] Input validation 加強 (50MB 檔案大小上限 + DICOM 副檔名檢查)
- [x] 本地開發環境 (docker-compose.yml + start.bat)
  - docker-compose: API (:8000) + Frontend (:3000) + PostgreSQL (:5432)
  - start.bat: Windows 一鍵啟動，auto venv + pip + uvicorn + http.server
  - 前端 API_BASE 自動偵測 (:3000 → :8000)
- [x] 測試環境 dev-ai-x-ray-detection.avision-gb10.org (NodePort 30081, 獨立 K8S deployments)
- [x] CI/CD Pipeline (GitHub Actions)
  - Lint: ruff check
  - Test: pytest 29 tests × Python 3.10/3.11/3.12 (矩陣測試)
  - Docker Build: Frontend (nginx:alpine) + API (python:3.12-slim) 驗證
  - pip cache + Docker layer cache 加速
  - push/PR to main, dev 自動觸發
- [x] 單元測試 (36 pytest, mock ONNX session, DICOM + predict + gradcam + auth + rate limiting + edge cases)

## 暫緩 ⏸️

- [ ] GPU 推論部署 (CPU 90ms 已夠快)
  - ✅ ONNX Runtime GPU wheel 已建 (v1.24.4, CUDA EP)
  - ✅ GPU Docker image 已建 (ai-xray-api:gpu, 598MB)
  - ❌ nvidia-device-plugin 無法在 minikube containerd 註冊 GPU resource
  - 待解: `/usr/local/nvidia/` 路徑不存在 (GB10 libs 在 `/usr/lib/aarch64-linux-gnu/`)
  - 替代方案: direct device mount / CDI / containerd nvidia runtime 設定

## 規劃中 📋

### 🔥 高優先

- [x] API Key 認證 + Rate limiting
  - Header-based `X-API-Key`，env `API_KEYS` 設定 (opt-in，未設定 = 關閉)
  - Rate limiting: 100 req/min per key（記憶體滑動窗口）
  - `/health` 永遠公開
- [x] CORS 白名單
  - env `CORS_ORIGINS` 逗號分隔，預設 `["*"]`
  - 允許: `ai-x-ray-detection.avision-gb10.org`, `dev-ai-x-ray-detection.avision-gb10.org`, `localhost`
- [x] 36 單元測試 (predict, gradcam, DICOM, auth, rate limiting, edge cases)
- [x] CI/CD Pipeline (GitHub Actions: lint + test × 3 + docker build)
- [x] UI 判讀流程優化 — 判讀後自動顯示原圖 + 熱力圖並排
- [x] PDF 報告重寫 — Canvas 中文渲染 + 影像原比例 + 彩色條狀圖
- [x] Dev URL 更名 → `dev-ai-x-ray-detection.avision-gb10.org`

### 📦 中優先

- [ ] 前端 UI 優化
  - 平板適配 (768px–1024px breakpoint)
  - 歷史記錄分頁 + 篩選
  - 深色/淺色主題切換
- [ ] 批次預測 API endpoint
  - POST /api/predict/batch (multipart, 多檔一次上傳)
  - 目前前端批次是逐張呼叫，server 端批次可減少 overhead
- [ ] Merge dev → main + 更新正式站
  - DICOM, RWD, PDF, batch upload 部署到 production
  - 重建 frontend Docker image (含最新 index.html)

### 🔮 長期

- [ ] 多模型比較 (ResNet, EfficientNet)
  - 需重新訓練 + ONNX 匯出
  - 前端模型選擇器 + 比較頁面
- [ ] Ensemble 推論
  - 多模型投票/加權平均
  - 依賴多模型完成
- [ ] 自動化模型 retraining pipeline
  - 資料標注回饋迴路
  - 定期 retrain + 效能比較
  - 模型版本管理
