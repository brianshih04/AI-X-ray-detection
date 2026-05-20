# TODO — 待辦事項

> 最後更新：2026-05-21

---

## 🔴 核心功能（必須完成）

### 1. Cloudflare Tunnel 外部存取
- **狀態**：DNS + cloudflared config 已設好，但外部測試回 404
- **問題**：cloudflared ingress rule 的 hostname 可能未正確 match
- **目標**：`curl https://ai-x-ray-detection.avision-gb10.org/health` 回傳 `{"status":"ok"}`
- **步驟**：
  - [ ] SSH 到 GB10 檢查 cloudflared logs（`journalctl -u cloudflared`）
  - [ ] 確認 tunnel route 已在 Cloudflare 註冊（可能需跑 `cloudflared tunnel route dns`）
  - [ ] 修復 hostname match 問題
  - [ ] 驗證外部 `/health` + `/api/predict` 都能通

### 2. 前端網頁
- **狀態**：未開始
- **目標**：使用者可上傳 X 光片、即時看到 15 類疾病信心分數
- **步驟**：
  - [ ] 選擇框架（React / Vue / 純 HTML+JS）
  - [ ] 實作上傳介面（拖拽或點選上傳 PNG/JPG）
  - [ ] 顯示推論結果（15 類 label + confidence bar chart）
  - [ ] 錯誤處理（檔案格式、大小限制、API 連線失敗）
  - [ ] RWD 響應式設計（手機也能用）
  - [ ] 部署至 Cloudflare（或 K8S 內）

### 3. 資料庫部署
- **狀態**：schema + Alembic migration + seed script 都寫好了，但沒部署
- **目標**：PostgreSQL 跑在 K8S 上，API 可讀寫
- **步驟**：
  - [ ] 寫 PostgreSQL K8S manifest（StatefulSet + PVC + Service）
  - [ ] 在 GB10 minikube 上部署 PostgreSQL
  - [ ] 跑 Alembic migration 建表
  - [ ] （選配）跑 `seed_nih.py` 匯入 108K 筆元資料
  - [ ] API 連線 DB 測試

### 4. 完整版 API 上線
- **狀態**：目前 K8S 跑簡化版（`api_build/`），完整版（`api/`）有 DB + Auth 但沒部署
- **目標**：K8S 跑完整版 API，支援 JWT 認證 + Patient/Image/Prediction CRUD
- **步驟**：
  - [ ] 更新 `api/Dockerfile`（加入 DB dependency）
  - [ ] 設定環境變數（DATABASE_URL, JWT_SECRET_KEY 等）
  - [ ] 建 Docker image 並部署到 minikube
  - [ ] 驗證所有 endpoints（auth, predict, patients, images, predictions）
  - [ ] Swagger UI (`/docs`) 可正常操作

### 5. Ingress Controller
- **狀態**：目前用 NodePort (30800) + cloudflared 直接指過去
- **目標**：用 Nginx Ingress 管理 routing，為未來多服務做準備
- **步驟**：
  - [ ] 啟用 minikube ingress addon（`minikube addons enable ingress`）
  - [ ] 寫 Ingress manifest（host: ai-x-ray-detection.avision-gb10.org）
  - [ ] 更新 cloudflared config 指向 Ingress（而非直接 NodePort）
  - [ ] 驗證外部存取正常

---

## 🟡 有 code 但沒整合

### 6. JWT 認證系統
- **狀態**：`api/src/api/auth.py` + `api/src/services/auth.py` 已完成
- **需要**：部署完整版 API + DB 後才能啟用
- [ ] 確認 register / login / me 三個 endpoint 正常
- [ ] 前端加入登入/註冊介面
- [ ] predict endpoint 加上可選認證

### 7. Patient / Image / Prediction CRUD
- **狀態**：`api/src/api/core.py` 已實作完整 CRUD endpoints
- **需要**：DB 上線後才能使用
- [ ] 驗證 patients 列表、詳情、篩選
- [ ] 驗證 images 綁定 patient
- [ ] 驗證 predictions 查詢

### 8. DICOM 格式支援
- **狀態**：README 有提到但沒實作
- [ ] 加入 `pydicom` dependency
- [ ] `preprocessing.py` 加入 DICOM → numpy 轉換
- [ ] 測試 DICOM 檔案上傳

---

## 🟢 Nice to have

### 9. GradCAM 視覺化
- [ ] 整合 `pytorch-grad-cam`
- [ ] API endpoint 回傳熱力圖（overlay 在原圖上）
- [ ] 前端顯示熱力圖

### 10. 批次推論 API
- [ ] `POST /api/predict/batch` 接受多張影像
- [ ] 非同步處理 + 回傳 batch job ID
- [ ] 輪詢或 WebSocket 回傳結果

### 11. CI/CD Pipeline
- [ ] GitHub Actions：lint + test on PR
- [ ] 自動建 Docker image + push to registry
- [ ] 自動 deploy to K8S on main merge

### 12. 模型監控
- [ ] Prometheus metrics（推論延遲、QPS、錯誤率）
- [ ] Grafana dashboard
- [ ] 模型 drift 偵測

### 13. EfficientNet-B3 / ViT 模型實驗
- [ ] 訓練 EfficientNet-B3 對比 DenseNet-121
- [ ] 訓練 ViT (Vision Transformer)
- [ ] 比較 AUROC / F1 / 推論速度

### 14. 多語系支援
- [ ] 前端 i18n（中文 / 英文）
- [ ] API 錯誤訊息多語系

---

## 📊 專案進度總覽

| 模組 | 進度 | 備註 |
|------|------|------|
| 資料前處理 (data_etl) | ✅ 100% | NIH CSV + Patient-level split |
| 模型訓練 (model_training) | ✅ 100% | DenseNet-121, AUROC 0.812 |
| ONNX 匯出 | ✅ 100% | opset 17, 28.1MB |
| 簡化版 API (api_build) | ✅ 100% | 已部署 K8S, 15 類推論 |
| 完整版 API (api) | 🟡 80% | Code 完成，缺 DB + 部署 |
| 資料庫 (database) | 🟡 70% | Schema 完成，缺 K8S 部署 |
| K8S 部署 (k8s) | 🟡 60% | 簡化版上線，缺 Ingress |
| 外部存取 (Cloudflare) | 🔴 90% | 設定完成，404 待修 |
| 前端網頁 (frontend) | 🔴 0% | 未開始 |
| JWT 認證 | 🟡 80% | Code 完成，缺整合 |
| DICOM 支援 | 🔴 0% | 未開始 |
| CI/CD | 🔴 0% | 未開始 |
