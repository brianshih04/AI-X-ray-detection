#!/usr/bin/env bash
# =============================================================================
# NIH ChestX-ray14 自動下載腳本
# =============================================================================
# 用法：
#   bash scripts/01_download.sh [--source kaggle|tfds|huggingface]
#                               [--data-dir ./data/raw]
#                               [--check-integrity]
#
# 前置需求：
#   - kaggle: pip install kaggle && mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/
#   - tfds:   pip install tensorflow-datasets
#   - huggingface: pip install datasets huggingface_hub
# =============================================================================

set -euo pipefail

SOURCE="${1:-kaggle}"
DATA_DIR="${2:-./data/raw}"
CHECK_INTEGRITY="${3:-}"
LOG_FILE="${DATA_DIR}/download.log"

# 14 個官方標籤
ALL_LABELS=(
    "Atelectasis" "Cardiomegaly" "Effusion" "Infiltration"
    "Mass" "Nodule" "Pneumonia" "Pneumothorax"
    "Emphysema" "Fibrosis" "Pleural_Thickening" "Consolidation"
    "Edema" "Hernia"
)

mkdir -p "${DATA_DIR}"
echo "[$(date)] Starting download from ${SOURCE}" | tee -a "${LOG_FILE}"

case "${SOURCE}" in

  kaggle)
    echo "[DOWNLOAD] Kaggle source selected" | tee -a "${LOG_FILE}"

    if ! command -v kaggle &>/dev/null; then
      echo "ERROR: kaggle CLI not found. Install with: pip install kaggle" | tee -a "${LOG_FILE}"
      exit 1
    fi

    # 檢查是否已有压好的檔案
    if [[ -d "${DATA_DIR}/nih-chest-xrays/images" ]] && \
       [[ -f "${DATA_DIR}/nih-chest-xrays/Data_Entry_2017_v2020.csv" ]]; then
      echo "[SKIP] Dataset already exists at ${DATA_DIR}/nih-chest-xrays/" | tee -a "${LOG_FILE}"
    else
      echo "[DOWNLOAD] Downloading from Kaggle..." | tee -a "${LOG_FILE}"
      kaggle datasets download -d nih-chest-xrays \
        -p "${DATA_DIR}" \
        --unzip \
        --quiet 2>&1 | tee -a "${LOG_FILE}"
      echo "[DONE] Download complete" | tee -a "${LOG_FILE}"
    fi
    ;;

  tfds)
    echo "[DOWNLOAD] TensorFlow Datasets source selected" | tee -a "${LOG_FILE}"

    if ! python3 -c "import tensorflow_datasets as tfds" 2>/dev/null; then
      echo "Installing tensorflow-datasets..." | tee -a "${LOG_FILE}"
      pip install tensorflow-datasets -q
    fi

    DS_DIR="${DATA_DIR}/tfds_nih"
    mkdir -p "${DS_DIR}"

    python3 - <<'PYEOF' 2>&1 | tee -a "${LOG_FILE}"
import os, tensorflow_datasets as tfds

ds_dir = os.environ.get("DS_DIR", "./data/raw/tfds_nih")
os.makedirs(ds_dir, exist_ok=True)

print(f"Downloading NIH ChestX-ray to {ds_dir} ...")
# 一次下載全量 train split
ds, info = tfds.load(
    "nih_chest_xray",
    split="train",
    data_dir=ds_dir,
    with_info=True,
    download=True,
)
print(f"Download complete. Dataset info: {info}")
PYEOF
    ;;

  huggingface)
    echo "[DOWNLOAD] HuggingFace source selected" | tee -a "${LOG_FILE}"

    if ! python3 -c "import datasets" 2>/dev/null; then
      pip install datasets huggingface_hub -q
    fi

    HF_DIR="${DATA_DIR}/hf_nih"
    mkdir -p "${HF_DIR}"

    python3 - <<'PYEOF' 2>&1 | tee -a "${LOG_FILE}"
import os, datasets

hf_dir = os.environ.get("HF_DIR", "./data/raw/hf_nih")
os.makedirs(hf_dir, exist_ok=True)

print(f"Downloading NIH ChestX-ray from HuggingFace to {hf_dir} ...")
ds = datasets.load_dataset(
    "LiuHeng0621/ChestXray14",
    split="train",
    cache_dir=hf_dir,
    trust_remote_code=True,
)
print(f"Download complete. Columns: {ds.column_names}")
PYEOF
    ;;

  *)
    echo "ERROR: Unknown source '${SOURCE}'. Use: kaggle | tfds | huggingface" | tee -a "${LOG_FILE}"
    exit 1
    ;;
esac

# ── 資料完整性驗證 ──────────────────────────────────────────────────────────
if [[ "${CHECK_INTEGRITY:-}" == "--check-integrity" ]] || [[ -d "${DATA_DIR}/nih-chest-xrays" ]]; then
  echo "" | tee -a "${LOG_FILE}"
  echo "[INTEGRITY] Checking dataset..." | tee -a "${LOG_FILE}"

  EXPECTED_IMAGES=108948
  EXPECTED_PATIENTS=32717
  EXPECTED_COLUMNS=("Image Index" "Finding Labels" "Patient ID" "Patient Age" "Patient Gender" "View Position")

  if [[ -f "${DATA_DIR}/nih-chest-xrays/Data_Entry_2017_v2020.csv" ]]; then
    CSV="${DATA_DIR}/nih-chest-xrays/Data_Entry_2017_v2020.csv"
  elif [[ -f "${DATA_DIR}/Data_Entry_2017_v2020.csv" ]]; then
    CSV="${DATA_DIR}/Data_Entry_2017_v2020.csv"
  else
    echo "[WARN] CSV not found, skipping CSV check" | tee -a "${LOG_FILE}"
    CSV=""
  fi

  if [[ -n "${CSV}" ]]; then
    python3 - <<'PYEOF' 2>&1 | tee -a "${LOG_FILE}"
import csv, os, sys

csv_path = os.environ.get("CSV", "")
if not csv_path or not os.path.exists(csv_path):
    print("[SKIP] CSV not found")
    sys.exit(0)

expected_cols = os.environ.get("EXPECTED_COLS", "").split(",")
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"  CSV rows: {len(rows)}")

# 檢查欄位
missing_cols = [c for c in expected_cols if c and c not in rows[0]] if expected_cols else []
if missing_cols:
    print(f"  [WARN] Missing columns: {missing_cols}")
else:
    print(f"  CSV columns OK: {list(rows[0].keys())}")

# 基本統計
import collections
patients = set()
labels_all = collections.Counter()
no_finding = 0
for r in rows:
    patients.add(int(r["Patient ID"]))
    lbls = r["Finding Labels"]
    if lbls == "No Finding":
        no_finding += 1
    else:
        for l in lbls.split("|"):
            labels_all[l.strip()] += 1

print(f"  Unique patients: {len(patients)}")
print(f"  No Finding: {no_finding} ({no_finding/len(rows)*100:.1f}%)")
print(f"  Has findings: {len(rows)-no_finding}")
print(f"  Top-5 labels:")
for lbl, cnt in labels_all.most_common(5):
    print(f"    {lbl}: {cnt} ({cnt/len(rows)*100:.1f}%)")
PYEOF
  fi

  # 圖片數量抽檢
  IMG_DIR=""
  for d in "${DATA_DIR}/nih-chest-xrays/images" "${DATA_DIR}/images"; do
    if [[ -d "${d}" ]]; then IMG_DIR="${d}"; break; fi
  done

  if [[ -n "${IMG_DIR}" && -d "${IMG_DIR}" ]]; then
    IMG_COUNT=$(find "${IMG_DIR}" -name "*.png" | wc -l)
    echo "[INTEGRITY] PNG images found: ${IMG_COUNT} (expected: ${EXPECTED_IMAGES})" | tee -a "${LOG_FILE}"
    if [[ "${IMG_COUNT}" -eq "${EXPECTED_IMAGES}" ]]; then
      echo "[OK] Image count matches expected" | tee -a "${LOG_FILE}"
    else
      echo "[WARN] Image count mismatch — expected ${EXPECTED_IMAGES}, found ${IMG_COUNT}" | tee -a "${LOG_FILE}"
    fi
  else
    echo "[WARN] images/ directory not found at expected location" | tee -a "${LOG_FILE}"
  fi
fi

echo "" | tee -a "${LOG_FILE}"
echo "[COMPLETE] Download & integrity check done at $(date)" | tee -a "${LOG_FILE}"
echo "Dataset root: ${DATA_DIR}"
