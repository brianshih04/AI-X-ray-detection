@echo off
chcp 65001 >nul
echo ========================================
echo   🫁 胸腔 X 光片自動判讀系統
echo   AI X-ray Detection - Local Setup
echo ========================================
echo.

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 找不到 Python，請先安裝 Python 3.12+
    echo    https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if in venv
if defined VIRTUAL_ENV (
    echo ✅ 虛擬環境: %VIRTUAL_ENV%
) else (
    echo 📦 建立虛擬環境...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo ✅ 虛擬環境已啟動
)

REM Install dependencies
echo 📥 安裝依賴套件...
pip install -r api_build_onnx\requirements.txt -q

REM Start API server
echo.
echo 🚀 啟動 API 伺服器 (http://localhost:8000)
echo 📁 前端頁面 (http://localhost:3000)
echo.
echo 按 Ctrl+C 停止伺服器
echo ========================================

start "" python -m http.server 3000 --directory frontend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --app-dir api_build_onnx
