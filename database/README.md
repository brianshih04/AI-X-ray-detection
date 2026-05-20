# ChestXpert Database Layer

> 專案資料庫層：PostgreSQL + SQLAlchemy ORM + Alembic migration + NIH seed script

## 架構概覽

```
t_0fc6a67b/
├── ERD.md                      # 完整 ERD 文件（表格規格、關係圖、備份策略）
├── config.py                   # Pydantic Settings 環境變數管理
├── requirements.txt            # Python 依賴
├── .env.example                # 環境變數範例
│
├── src/
│   ├── database.py             # SQLAlchemy async/sync engines + session
│   ├── models.py               # ORM models (User, Patient, Image, ImageLabel, Prediction)
│   ├── schemas/                # Pydantic schemas (API 請求/回應)
│   │   ├── user.py
│   │   ├── patient.py
│   │   ├── image.py
│   │   └── prediction.py
│   └── crud/                   # CRUD 操作層
│       ├── crud_utils.py       # CRUDBase 泛型類
│       ├── user.py
│       ├── patient.py
│       ├── image.py
│       └── prediction.py
│
├── alembic/                    # Alembic migration 管理
│   ├── env.py                  # Migration 環境配置
│   ├── script.py.mako          # Migration 模板
│   └── versions/
│       └── 001_initial.py     # 初始 schema migration
│
├── scripts/
│   └── seed_nih.py             # NIH CSV seed script (108K images)
│
└── tests/                      # 測試（預留）
```

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 環境變數

```bash
cp .env.example .env
# 編輯 .env 填入實際值
```

### 3. 執行 Migration

```bash
# 初始化版本表
alembic upgrade head
```

### 4. Seed NIH 資料

```bash
python scripts/seed_nih.py \
    --csv /path/to/Data_Entry_2017_v2020.csv \
    --images-dir /path/to/nih-chest-xrays/images
```

## 五張核心表格

| 表格 | 用途 |
|------|------|
| `users` | 醫師/放射科醫師/管理者認證 |
| `patients` | 病患記錄（連結 NIH Patient ID） |
| `images` | X 光影像記錄（連結檔案路徑） |
| `image_labels` | NLP 挖掘的疾病標籤（來自 NIH 報告） |
| `predictions` | 模型預測結果（多版本支援） |

## API Contract（供 T7 前端對接）

### 核心 Entity Schema

| Entity | 主要欄位 |
|--------|---------|
| Patient | id, patient_id_ext, age, gender, user_id |
| Image | id, patient_id, image_index, file_path, view_position |
| Prediction | id, image_id, model_version, label_name, confidence, is_approved |

> 完整 API schema 定義在 `src/schemas/`。T7 前端應先對照 `ERD.md` 的欄位定義與 `src/models.py` 的 constraint。

## 備份策略

- **每日 Full Backup** (`pg_dump -Fc`)，保留 30 天
- **每小時 WAL archiving**，支援 PITR
- 詳細策略見 `ERD.md#Backup-Strategy`

## 索引起略（見 `ERD.md#Indexing-Strategy`）

- Primary indexes：定義於 schema
- Secondary indexes：患者查詢、預測分析、模型版本篩選
- Partial indexes：已核准預測、近期預測

## 與 T9（FastAPI 後端）的銜接

本層完成後，T9 可直接使用：
- `src/crud/` — 所有 CRUD 操作（async）
- `src/schemas/` — Pydantic schemas 做 request/response validation
- `src/models.py` — ORM models 供 SQLAlchemy 查詢
- `src/database.py` — `get_async_session()` 注入 FastAPI dependency

```python
# T9 FastAPI 示例
from src.database import get_async_session
from src.crud import patient_crud

@app.get("/api/patients/{patient_id}")
async def get_patient(patient_id: UUID, db: AsyncSession = Depends(get_async_session)):
    patient = await patient_crud.get(db, patient_id)
    if not patient:
        raise HTTPException(404, "Patient not found")
    return patient
```