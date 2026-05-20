#!/usr/bin/env bash
# =============================================================================
# NIH ChestX-ray14 — 完整 ETL Pipeline
# =============================================================================
# 一鍵執行：下載 → 清洗 → split → 驗證
#
# 用法：
#   bash scripts/run_etl.sh                    # 全流程
#   bash scripts/run_etl.sh --skip-download    # 跳過下載
#   bash scripts/run_etl.sh --skip-validate    # 跳過驗證（只需清洗）
#   bash scripts/run_etl.sh --clahe            # 啟用 CLAHE
#   bash scripts/run_etl.sh --size 256         # 使用 256x256
#
# 環境變數：
#   DATA_DIR      原始資料目錄（預設 ./data/raw）
#   PROCESSED_DIR 清洗後資料目錄（預設 ./data/processed）
#   IMG_DIR       影像目錄（預設 ${DATA_DIR}/nih-chest-xrays/images）
#   SEED          split seed（預設 42）
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── 預設參數 ─────────────────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-${WORKSPACE_DIR}/data/raw}"
PROCESSED_DIR="${PROCESSED_DIR:-${WORKSPACE_DIR}/data/processed}"
IMG_DIR="${IMG_DIR:-${DATA_DIR}/nih-chest-xrays/images}"
SEED="${SEED:-42}"
SIZE=224
SKIP_DOWNLOAD=false
SKIP_VALIDATE=false
USE_CLAHE=false
DOWNLOAD_SOURCE="kaggle"

# ── 解析參數 ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-download)  SKIP_DOWNLOAD=true ;;
    --skip-validate)  SKIP_VALIDATE=true ;;
    --clahe)          USE_CLAHE=true ;;
    --size)           SIZE="$2"; shift ;;
    --seed)           SEED="$2"; shift ;;
    --source)         DOWNLOAD_SOURCE="$2"; shift ;;
    --data-dir)       DATA_DIR="$2"; shift ;;
    --img-dir)        IMG_DIR="$2"; shift ;;
    --help|-h)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --skip-download   Skip download step"
      echo "  --skip-validate   Skip validation step"
      echo "  --clahe           Enable CLAHE contrast enhancement"
      echo "  --size N          Image size (default: 224)"
      echo "  --seed N          Split seed (default: 42)"
      echo "  --source kaggle|tfds|huggingface"
      echo "  --data-dir DIR    Raw data directory"
      echo "  --img-dir DIR     Image directory"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
  shift
done

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  NIH ChestX-ray14 — ETL Pipeline                        ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Data dir:      ${DATA_DIR}"
echo "║  Processed dir: ${PROCESSED_DIR}"
echo "║  Image dir:     ${IMG_DIR}"
echo "║  Image size:    ${SIZE}x${SIZE}"
echo "║  Seed:          ${SEED}"
echo "║  CLAHE:         ${USE_CLAHE}"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

PASS=0
FAIL=0

# ── Step 1: Download ────────────────────────────────────────────────────────
if [[ "${SKIP_DOWNLOAD}" == "true" ]]; then
    echo "[1/4] [SKIP] Download (as requested)"
else
    echo "[1/4] Downloading dataset..."
    if bash "${SCRIPT_DIR}/01_download.sh" "${DOWNLOAD_SOURCE}" "${DATA_DIR}" --check-integrity; then
        echo "  ✅ Download complete"
        PASS=$((PASS + 1))
    else
        echo "  ❌ Download failed"
        FAIL=$((FAIL + 1))
    fi
fi

# ── Step 2: Clean & Split ───────────────────────────────────────────────────
echo ""
echo "[2/4] Cleaning metadata & splitting..."
CSV_FILE="${DATA_DIR}/nih-chest-xrays/Data_Entry_2017_v2020.csv"
if [[ ! -f "${CSV_FILE}" ]]; then
    # 嘗試其他可能位置
    for alt in \
        "${DATA_DIR}/Data_Entry_2017_v2020.csv" \
        "${DATA_DIR}/nih-chest-xrays/Data_Entry_2017.csv" \
        "${DATA_DIR}/Data_Entry_2017.csv"; do
        if [[ -f "${alt}" ]]; then
            CSV_FILE="${alt}"
            break
        fi
    done
fi

if [[ ! -f "${CSV_FILE}" ]]; then
    echo "  ❌ CSV not found. Place Data_Entry_2017_v2020.csv in ${DATA_DIR}/"
    FAIL=$((FAIL + 1))
else
    mkdir -p "${PROCESSED_DIR}"
    if python3 "${SCRIPT_DIR}/02_clean_metadata.py" \
        --csv "${CSV_FILE}" \
        --img-dir "${IMG_DIR}" \
        --out-dir "${PROCESSED_DIR}" \
        --seed "${SEED}"; then
        echo "  ✅ Clean & split complete"
        PASS=$((PASS + 1))
    else
        echo "  ❌ Clean & split failed"
        FAIL=$((FAIL + 1))
    fi
fi

# ── Step 3: Validate ────────────────────────────────────────────────────────
if [[ "${SKIP_VALIDATE}" == "true" ]]; then
    echo ""
    echo "[3/4] [SKIP] Validation (as requested)"
    PASS=$((PASS + 1))
else
    echo ""
    echo "[3/4] Validating data..."
    if python3 "${SCRIPT_DIR}/03_validate_data.py" \
        --data-dir "${PROCESSED_DIR}" \
        --img-dir "${IMG_DIR}" \
        --size "${SIZE}"; then
        echo "  ✅ Validation passed"
        PASS=$((PASS + 1))
    else
        echo "  ❌ Validation failed"
        FAIL=$((FAIL + 1))
    fi
fi

# ── Step 4: Unit Tests ──────────────────────────────────────────────────────
echo ""
echo "[4/4] Running unit tests..."
cd "${WORKSPACE_DIR}"
if python3 -m pytest tests/test_etl.py -v --tb=short 2>&1; then
    echo "  ✅ Unit tests passed"
    PASS=$((PASS + 1))
else
    echo "  ❌ Unit tests failed"
    FAIL=$((FAIL + 1))
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  ETL Pipeline Summary"
echo "══════════════════════════════════════════════════"
echo "  Passed: ${PASS}"
echo "  Failed: ${FAIL}"
if [[ ${FAIL} -eq 0 ]]; then
    echo "  ✅ ALL STEPS PASSED"
    echo ""
    echo "  Output files:"
    echo "    ${PROCESSED_DIR}/train.csv"
    echo "    ${PROCESSED_DIR}/val.csv"
    echo "    ${PROCESSED_DIR}/test.csv"
    echo "    ${PROCESSED_DIR}/class_weights.json"
    echo "    ${PROCESSED_DIR}/metadata.json"
    echo ""
    echo "  Next step: Use src/chestxray14_dataset.py to build DataLoaders"
    echo "    from chestxray14_dataset import build_dataloaders"
    echo "    loaders = build_dataloaders('${PROCESSED_DIR}', '${IMG_DIR}')"
else
    echo "  ❌ SOME STEPS FAILED — review output above"
    exit 1
fi
