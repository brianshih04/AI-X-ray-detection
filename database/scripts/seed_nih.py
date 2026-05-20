#!/usr/bin/env python3
"""
NIH ChestX-ray14 CSV → PostgreSQL seed script.

Usage:
    python scripts/seed_nih.py --csv Data_Entry_2017_v2020.csv --images-dir ./images/

This script:
1. Parses the NIH CSV metadata
2. Upserts patients (by patient_id_ext)
3. Upserts images (by image_index)
4. Upserts image_labels (NLP-mined disease labels)
5. Handles ~109K images in batches for memory efficiency

Expected CSV columns:
    - Image Index         : filename (e.g. 00000001_000.png)
    - Finding Labels      : pipe-separated labels (e.g. "Atelectasis|Effusion" or "No Finding")
    - Follow-up #         : follow-up number
    - Patient ID          : NIH patient ID
    - Patient Age         : age in years
    - Patient Gender      : M or F
    - View Position       : AP or PA
    - OriginalImage[Width|Height] : DICOM original dimensions
    - OriginalImagePixelSpacing[x|y] : pixel spacing in mm
"""
from __future__ import annotations

import argparse
import csv
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import IO, Iterator

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import text
from sqlalchemy.orm import Session

from config import db_settings
from database import sync_engine, SyncSessionLocal
from src.models import Image, ImageLabel, Patient, LABEL_SET

# Batch size for bulk operations
BATCH_SIZE = 1000

# Valid NIH labels
VALID_LABELS = list(LABEL_SET)


def parse_csv(csv_path: str) -> Iterator[dict]:
    """Parse NIH CSV file and yield row dicts."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def parse_labels(finding_labels_str: str) -> list[str]:
    """Parse pipe-separated finding labels."""
    if not finding_labels_str or finding_labels_str.strip() == "":
        return []
    if finding_labels_str == "No Finding":
        return []
    labels = [l.strip() for l in finding_labels_str.split("|")]
    return [l for l in labels if l in LABEL_SET]


def upsert_patient(
    session: Session,
    patient_id_ext: str,
    age: int,
    gender: str,
) -> uuid.UUID:
    """Upsert patient and return UUID."""
    # Check if exists
    result = session.execute(
        text("SELECT id FROM patients WHERE patient_id_ext = :pid"),
        {"pid": patient_id_ext},
    )
    row = result.fetchone()

    if row:
        return row[0]

    # Create new
    patient_id = uuid.uuid4()
    now = datetime.utcnow()
    session.execute(
        text("""
            INSERT INTO patients (id, patient_id_ext, age, gender, created_at, updated_at)
            VALUES (:id, :pid_ext, :age, :gender, :now, :now)
        """),
        {
            "id": patient_id,
            "pid_ext": patient_id_ext,
            "age": age,
            "gender": gender,
            "now": now,
        },
    )
    return patient_id


def upsert_image(
    session: Session,
    patient_uuid: uuid.UUID,
    image_index: str,
    image_dir: Path,
    row: dict,
) -> tuple[uuid.UUID, bool]:
    """
    Upsert image record.

    Returns: (image_uuid, is_new)
    """
    result = session.execute(
        text("SELECT id FROM images WHERE image_index = :idx"),
        {"idx": image_index},
    )
    row_img = result.fetchone()

    if row_img:
        return row_img[0], False

    image_id = uuid.uuid4()
    now = datetime.utcnow()

    # Parse optional DICOM metadata
    orig_w = _parse_int(row.get("OriginalImage[Width]"))
    orig_h = _parse_int(row.get("OriginalImage[Height]"))
    ps_x = _parse_float(row.get("OriginalImagePixelSpacing[x]"))
    ps_y = _parse_float(row.get("OriginalImagePixelSpacing[y]"))

    # Parse view position
    view_pos = row.get("View Position", "")
    if view_pos not in ("AP", "PA"):
        view_pos = None

    # File path
    file_path = str(image_dir / image_index)

    session.execute(
        text("""
            INSERT INTO images (
                id, patient_id, image_index, file_path,
                width, height, view_position,
                original_dcm_width, original_dcm_height,
                pixel_spacing_x, pixel_spacing_y,
                created_at
            ) VALUES (
                :id, :pid, :idx, :fp,
                1024, 1024, :vp,
                :odw, :odh,
                :psx, :psy,
                :now
            )
        """),
        {
            "id": image_id,
            "pid": patient_uuid,
            "idx": image_index,
            "fp": file_path,
            "vp": view_pos,
            "odw": orig_w,
            "odh": orig_h,
            "psx": ps_x,
            "psy": ps_y,
            "now": now,
        },
    )
    return image_id, True


def upsert_image_labels(
    session: Session,
    image_uuid: uuid.UUID,
    labels: list[str],
) -> int:
    """Upsert image labels (skip duplicates)."""
    now = datetime.utcnow()
    count = 0

    for label in labels:
        # Check if exists
        result = session.execute(
            text(
                "SELECT id FROM image_labels WHERE image_id = :iid AND label_name = :lbl"
            ),
            {"iid": image_uuid, "lbl": label},
        )
        if result.fetchone():
            continue

        label_id = uuid.uuid4()
        session.execute(
            text("""
                INSERT INTO image_labels (id, image_id, label_name, is_nlp_mined, created_at)
                VALUES (:id, :iid, :lbl, TRUE, :now)
            """),
            {"id": label_id, "iid": image_uuid, "lbl": label, "now": now},
        )
        count += 1

    return count


def _parse_int(val: str | None) -> int | None:
    """Safely parse integer."""
    if val is None or val.strip() == "":
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _parse_float(val: str | None) -> float | None:
    """Safely parse float."""
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def seed_nih(
    csv_path: str,
    image_dir: str,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Main seeding function."""
    image_dir = Path(image_dir)

    if not image_dir.exists():
        print(f"WARNING: Image directory not found: {image_dir}")
        print("         Image records will be created with non-verified paths.")

    print(f"Reading CSV: {csv_path}")
    csv_rows = parse_csv(csv_path)

    stats = {
        "total_rows": 0,
        "patients_created": 0,
        "patients_existed": 0,
        "images_created": 0,
        "images_existed": 0,
        "labels_created": 0,
        "skipped_invalid_labels": 0,
    }

    # Process in batches
    batch_patients: dict[str, uuid.UUID] = {}
    batch_images: dict[str, uuid.UUID] = {}
    batch_labels: dict[str, list[str]] = {}
    batch_rows: list[dict] = []

    def flush_batch():
        """Write accumulated batch to DB."""
        nonlocal stats, batch_patients, batch_images, batch_labels, batch_rows

        if not batch_rows:
            return

        with SyncSessionLocal() as session:
            with session.begin():
                for row in batch_rows:
                    pid_ext = row["Patient ID"]
                    image_index = row["Image Index"]

                    # Upsert patient
                    if pid_ext not in batch_patients:
                        patient_uuid = upsert_patient(
                            session,
                            pid_ext,
                            int(row["Patient Age"]),
                            row["Patient Gender"],
                        )
                        batch_patients[pid_ext] = patient_uuid
                        stats["patients_created"] += 1
                    else:
                        stats["patients_existed"] += 1

                    patient_uuid = batch_patients[pid_ext]

                    # Upsert image
                    if image_index not in batch_images:
                        image_uuid, is_new = upsert_image(
                            session,
                            patient_uuid,
                            image_index,
                            image_dir,
                            row,
                        )
                        batch_images[image_index] = image_uuid
                        if is_new:
                            stats["images_created"] += 1
                        else:
                            stats["images_existed"] += 1
                            continue  # Skip labels for existing images

                        # Parse and upsert labels
                        labels = parse_labels(row["Finding Labels"])
                        if labels:
                            n_labels = upsert_image_labels(session, image_uuid, labels)
                            stats["labels_created"] += n_labels

        # Reset batch
        batch_patients = {}
        batch_images = {}
        batch_labels = {}
        batch_rows = []

    total_rows = 0
    for row in parse_csv(csv_path):
        total_rows += 1
        batch_rows.append(row)
        stats["total_rows"] = total_rows

        if len(batch_rows) >= batch_size:
            print(f"\rProcessed {total_rows} rows...", end="", flush=True)
            flush_batch()

    # Flush remaining
    if batch_rows:
        print(f"\rProcessed {total_rows} rows...done", flush=True)
        flush_batch()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Seed NIH ChestX-ray14 data into PostgreSQL"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to Data_Entry_2017_v2020.csv",
    )
    parser.add_argument(
        "--images-dir",
        default="./data/nih-chest-xrays/images",
        help="Path to images directory",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count rows without inserting",
    )

    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: CSV file not found: {args.csv}")
        sys.exit(1)

    # Count rows first
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        row_count = sum(1 for _ in reader)
    print(f"CSV has {row_count:,} rows")

    if args.dry_run:
        print("Dry run — exiting without modifications")
        return

    print(f"Starting seed to {db_settings.sync_url}")
    print(f"Image directory: {args.images_dir}")
    print("-" * 50)

    stats = seed_nih(args.csv, args.images_dir, args.batch_size)

    print("\n" + "=" * 50)
    print("SEED COMPLETE")
    print("=" * 50)
    print(f"Total CSV rows    : {stats['total_rows']:,}")
    print(f"Patients created  : {stats['patients_created']:,}")
    print(f"Patients existed  : {stats['patients_existed']:,}")
    print(f"Images created    : {stats['images_created']:,}")
    print(f"Images existed   : {stats['images_existed']:,}")
    print(f"Labels created    : {stats['labels_created']:,}")
    print(f"Invalid labels    : {stats['skipped_invalid_labels']:,}")


if __name__ == "__main__":
    main()