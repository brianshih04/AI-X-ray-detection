# ChestXpert Database Schema

> Database schema for NIH ChestX-ray14 multi-label classification system.
> Designed to support T7 API integration.

## ERD Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     users       │     │    patients      │     │     images      │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ id (PK)         │     │ id (PK)         │     │ id (PK)         │
│ username        │◄────│ user_id (FK)    │     │ patient_id (FK) │
│ email           │     │ patient_id_ext  │◄────│ image_index     │
│ hashed_password │     │ age             │     │ file_path       │
│ is_active       │     │ gender          │     │ width           │
│ is_superuser    │     │ created_at      │     │ height          │
│ created_at      │     │ updated_at      │     │ view_position    │
│ updated_at      │     └─────────────────┘     │ original_dcm_w   │
└─────────────────┘             │              │ original_dcm_h   │
                                │              │ pixel_spacing_x  │
                                │              │ pixel_spacing_y  │
                                │              │ created_at       │
                                │              └────────┬────────┘
                                │                       │
                                ▼                       ▼
                    ┌─────────────────┐     ┌─────────────────┐
                    │   image_labels  │     │   predictions   │
                    ├─────────────────┤     ├─────────────────┤
                    │ id (PK)         │     │ id (PK)         │
                    │ image_id (FK)  │     │ image_id (FK)  │
                    │ label_name     │     │ model_version   │
                    │ is_nlp_mined   │     │ confidence      │
                    │ created_at      │     │ predicted_at    │
                    └─────────────────┘     │ created_by (FK) │
                                            │ user_id (FK)   │
                                            │ is_approved    │
                                            │ notes          │
                                            └─────────────────┘
```

## Table Specifications

### 1. users

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, DEFAULT uuid_generate_v4() | Primary key |
| username | VARCHAR(50) | UNIQUE, NOT NULL | Login username |
| email | VARCHAR(255) | UNIQUE, NOT NULL | Email address |
| hashed_password | VARCHAR(255) | NOT NULL | Bcrypt hashed password |
| is_active | BOOLEAN | DEFAULT TRUE | Account active flag |
| is_superuser | BOOLEAN | DEFAULT FALSE | Admin flag |
| created_at | TIMESTAMP | DEFAULT NOW() | Creation timestamp |
| updated_at | TIMESTAMP | DEFAULT NOW() | Last update |

**Indexes:**
- `idx_users_username` ON (username)
- `idx_users_email` ON (email)

### 2. patients

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, DEFAULT uuid_generate_v4() | Primary key |
| user_id (nullable) | UUID | FK → users.id | Assigned physician |
| patient_id_ext | VARCHAR(50) | UNIQUE, NOT NULL | External NIH Patient ID |
| age | INTEGER | NOT NULL, CHECK 0-150 | Patient age |
| gender | VARCHAR(1) | NOT NULL, CHECK (M/F) | Gender |
| created_at | TIMESTAMP | DEFAULT NOW() | Creation timestamp |
| updated_at | TIMESTAMP | DEFAULT NOW() | Last update |

**Indexes:**
- `idx_patients_external_id` ON (patient_id_ext)
- `idx_patients_user_id` ON (user_id)

**Constraints:**
- `chk_patients_age` CHECK (age >= 0 AND age <= 150)
- `chk_patients_gender` CHECK (gender IN ('M', 'F'))

### 3. images

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, DEFAULT uuid_generate_v4() | Primary key |
| patient_id | UUID | FK → patients.id, NOT NULL | Parent patient |
| image_index | VARCHAR(50) | UNIQUE, NOT NULL | NIH image filename |
| file_path | VARCHAR(500) | NOT NULL | Local storage path |
| width | INTEGER | DEFAULT 1024 | Image width |
| height | INTEGER | DEFAULT 1024 | Image height |
| view_position | VARCHAR(10) | CHECK (AP/PA/UNKNOWN) | X-ray view |
| original_dcm_width | INTEGER | | Original DICOM width |
| original_dcm_height | INTEGER | | Original DICOM height |
| pixel_spacing_x | FLOAT | | Pixel spacing X (mm) |
| pixel_spacing_y | FLOAT | | Pixel spacing Y (mm) |
| created_at | TIMESTAMP | DEFAULT NOW() | Creation timestamp |

**Indexes:**
- `idx_images_patient_id` ON (patient_id)
- `idx_images_image_index` ON (image_index)
- `idx_images_view_position` ON (view_position)

**Constraints:**
- `chk_images_view_position` CHECK (view_position IN ('AP', 'PA', 'UNKNOWN'))

### 4. image_labels (NLP-mined from NIH reports)

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, DEFAULT uuid_generate_v4() | Primary key |
| image_id | UUID | FK → images.id, NOT NULL | Parent image |
| label_name | VARCHAR(50) | NOT NULL | Disease label |
| is_nlp_mined | BOOLEAN | DEFAULT TRUE | From NLP text mining |
| created_at | TIMESTAMP | DEFAULT NOW() | Creation timestamp |

**Indexes:**
- `idx_image_labels_image_id` ON (image_id)
- `idx_image_labels_label_name` ON (label_name)
- `idx_image_labels_composite` ON (image_id, label_name) UNIQUE

**Note:** Each (image_id, label_name) pair is unique — multi-label handled at query level.

### 5. predictions

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | UUID | PK, DEFAULT uuid_generate_v4() | Primary key |
| image_id | UUID | FK → images.id, NOT NULL | Predicted image |
| model_version | VARCHAR(50) | NOT NULL | Model identifier |
| label_name | VARCHAR(50) | NOT NULL | Predicted disease |
| confidence | FLOAT | NOT NULL, CHECK 0-1 | Prediction confidence |
| predicted_at | TIMESTAMP | NOT NULL | Prediction timestamp |
| created_by | UUID | FK → users.id | Creator user |
| is_approved | BOOLEAN | DEFAULT FALSE | Approved by physician |
| notes | TEXT | | Physician notes |

**Indexes:**
- `idx_predictions_image_id` ON (image_id)
- `idx_predictions_model_version` ON (model_version)
- `idx_predictions_created_by` ON (created_by)
- `idx_predictions_label_conf` ON (label_name, confidence)
- `idx_predictions_predicted_at` ON (predicted_at)

**Constraints:**
- `chk_predictions_confidence` CHECK (confidence >= 0.0 AND confidence <= 1.0)

## Relationships

```
users (1) ──── (N) patients (assigned physician)
              │
              │ (1)
              │
              ▼
         patients (1) ──── (N) images
              │                  │
              │                  │ (1)
              │                  ▼
              │           images (1) ──── (N) image_labels
              │                  │
              │                  │ (1)
              │                  ▼
              │           images (1) ──── (N) predictions
              │                                │
              │                                │ (0..1)
              │                                ▼
              └──────────────────────── users (1) (creator)
```

## NIH ChestX-ray14 Disease Labels

| Label Name | Description |
|------------|-------------|
| Atelectasis | 肺擴張不全 |
| Cardiomegaly | 心臟肥大 |
| Effusion | 胸腔積液 |
| Infiltration | 浸潤 (noisy, consider removing) |
| Mass | 腫塊 |
| Nodule | 結節 |
| Pneumonia | 肺炎 |
| Pneumothorax | 氣胸 |
| Emphysema | 肺氣腫 |
| Fibrosis | 肺纖維化 |
| Pleural_Thickening | 肋膜增厚 |
| Consolidation | 肺實質化 |
| Edema | 肺水腫 |
| Hernia | 橫膈疝氣 (0.2%, rarest) |

## Backup Strategy

### Backup Types

1. **Full Backup (Daily)**
   - `pg_dump -Fc chestxpert -f backup_$(date +%Y%m%d).dump`
   - Retention: 30 days

2. **Incremental Backup (Hourly via WAL)**
   - Configure PostgreSQL WAL archiving
   - Point-in-time recovery enabled

3. **Point-in-Time Recovery (PITR)**
   - WAL archive to S3-compatible storage
   - RPO: 1 hour

### Backup Schedule

| Time | Type | Retention |
|------|------|-----------|
| 02:00 UTC | Full | 30 days |
| Every hour | WAL | 7 days |
| Weekly (Sunday) | Full | 12 weeks |

### Restoration

```bash
# Full restore
pg_restore -d chestxpert_backup backup_20240101.dump

# PITR restore
pg_restore --point-in-time='2024-01-01 15:30:00' backup.dump
```

## Indexing Strategy

### Primary Indexes (defined in schema)

See table specifications above.

### Secondary Indexes (for common queries)

```sql
-- Patient lookup by external ID
CREATE INDEX idx_patients_external_id ON patients(patient_id_ext);

-- Images by patient (for patient image list)
CREATE INDEX idx_images_patient_id ON images(patient_id);

-- Predictions by model version (for A/B testing)
CREATE INDEX idx_predictions_model_version ON predictions(model_version);

-- Predictions by date range (for dashboard)
CREATE INDEX idx_predictions_predicted_at ON predictions(predicted_at DESC);

-- Label search (for filtering)
CREATE INDEX idx_image_labels_label_name ON image_labels(label_name);

-- Composite: patient + date for timeline
CREATE INDEX idx_predictions_patient_date ON predictions(image_id, predicted_at DESC)
  INCLUDE (confidence, label_name);
```

### Partial Indexes

```sql
-- Only approved predictions (faster reads)
CREATE INDEX idx_predictions_approved ON predictions(image_id, label_name)
  WHERE is_approved = TRUE;

-- Only recent predictions (last 30 days)
CREATE INDEX idx_predictions_recent ON predictions(image_id, predicted_at DESC)
  WHERE predicted_at > NOW() - INTERVAL '30 days';
```

## Performance Considerations

1. **Batch inserts**: Use `COPY` for bulk NIH data import
2. **Connection pooling**: PgBouncer with pool_mode=transaction
3. **Read replicas**: Route analytics queries to replicas
4. **Partitioning** (future): Partition `predictions` by month for archival