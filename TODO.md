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
  - jsPDF CDN，標題 + 最高信心診斷 + 15 疾病條狀圖 + 免責聲明
- [x] 支援批次上傳 (多檔案選擇器，循序 API 呼叫，即時進度)
- [x] Input validation 加強 (50MB 檔案大小上限 + DICOM 副檔名檢查)
- [x] 本地開發環境 (docker-compose.yml + start.bat)
  - docker-compose: API (:8000) + Frontend (:3000) + PostgreSQL (:5432)
  - start.bat: Windows 一鍵啟動，auto venv + pip + uvicorn + http.server
  - 前端 API_BASE 自動偵測 (:3000 → :8000)
- [x] 測試環境 ai-xray-test.avision-gb10.org (NodePort 30081, 獨立 K8S deployments)

## 暫緩 ⏸️

- [ ] GPU 推論部署 (CPU 90ms 已夠快)
  - ✅ ONNX Runtime GPU wheel 已建 (v1.24.4, CUDA EP)
  - ✅ GPU Docker image 已建 (ai-xray-api:gpu, 598MB)
  - ❌ nvidia-device-plugin 無法在 minikube containerd 註冊 GPU resource
  - 待解: `/usr/local/nvidia/` 路徑不存在 (GB10 libs 在 `/usr/lib/aarch64-linux-gnu/`)
  - 替代方案: direct device mount / CDI / containerd nvidia runtime 設定

## 規劃中 📋

- [ ] 前端優化
  - 更多裝置適配 (平板)
  - 結果歷史記錄 UI 優化
- [ ] 模型改善
  - 多模型比較 (ResNet, EfficientNet)
  - Ensemble 推論
- [ ] API 完整版
  - API Key 認證
  - Rate limiting
  - 批次預測 endpoint
- [ ] CI/CD Pipeline
  - GitHub Actions: lint → test → build → deploy
  - 自動化模型 retraining pipeline
- [ ] 安全性
  - HTTPS 強制 (Cloudflare 已處理)
  - CORS 限制 (目前 allow_origins=["*"])
