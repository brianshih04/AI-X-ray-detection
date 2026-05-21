# TODO

## 已完成 ✅

- [x] 資料集準備 (NIH ChestX-ray14)
- [x] 資料預處理 (CSV 轉換, train/val/test split)
- [x] DenseNet-121 模型訓練 (Mean AUROC 0.812)
- [x] 模型評估 + ONNX 匯出
- [x] FastAPI 推論 API (ONNX Runtime, POST /api/predict)
- [x] 前端 UI (深色主題, 拖曳上傳, 14 疾病條狀圖)
- [x] Docker 容器化 (frontend 61.5MB + API 406MB)
- [x] K8S 部署 (Minikube v1.35.1, containerd)
- [x] Cloudflare Tunnel 外部存取
- [x] PostgreSQL 整合 (預測結果儲存)
- [x] E2E 測試通過 (CPU, ~90ms/張)
- [x] ONNX Runtime GPU wheel 建置 (aarch64 + sm_121, 54MB)
- [x] GPU Docker image 建置 (598MB)

## 進行中 🔧

- [ ] GPU 推論部署
  - ✅ ONNX Runtime GPU wheel 已建 (v1.24.4, CUDA EP)
  - ✅ GPU Docker image 已建 (ai-xray-api:gpu, 598MB)
  - ❌ nvidia-device-plugin 無法在 minikube containerd 註冊 GPU resource
  - 待解: `/usr/local/nvidia/` 路徑不存在 (GB10 libs 在 `/usr/lib/aarch64-linux-gnu/`)
  - 替代方案: direct device mount / CDI / containerd nvidia runtime 設定

## 規劃中 📋

- [ ] 前端優化
  - 手機版 RWD 微調
  - 支援批次上傳 (多張 X 光片)
  - 結果匯出 (PDF 報告)
  - 預測歷史記錄查詢頁面
- [ ] 模型改善
  - Grad-CAM 熱力圖 (視覺化模型關注區域)
  - 多模型比較 (ResNet, EfficientNet)
  - Ensemble 推論
- [ ] API 完整版
  - API Key 認證
  - Rate limiting
  - 批次預測 endpoint
  - 歷史查詢 API
- [ ] CI/CD Pipeline
  - GitHub Actions: lint → test → build → deploy
  - 自動化模型 retraining pipeline
- [ ] 安全性
  - HTTPS 強制 (Cloudflare 已處理)
  - Input validation 加強
