"""Initial schema: users, patients, images, image_labels, predictions

Revision ID: 001_initial
Revises:
Create Date: 2024-01-01 00:00:00

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# NIH ChestX-ray14 labels for constraint
VALID_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Consolidation",
    "Edema",
    "Hernia",
]

LABEL_LIST = ", ".join(f"'{l}'" for l in VALID_LABELS)


def upgrade() -> None:
    # ============================================================
    # Enable UUID extension
    # ============================================================
    op.execute("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\"")

    # ============================================================
    # Create users table
    # ============================================================
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("username", sa.String(50), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, default=True),
        sa.Column("is_superuser", sa.Boolean(), nullable=False, default=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username"),
        sa.UniqueConstraint("email"),
    )
    op.create_index("idx_users_username", "users", ["username"], unique=False)
    op.create_index("idx_users_email", "users", ["email"], unique=False)

    # ============================================================
    # Create patients table
    # ============================================================
    op.create_table(
        "patients",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("patient_id_ext", sa.String(50), nullable=False),
        sa.Column("age", sa.Integer(), nullable=False),
        sa.Column("gender", sa.String(1), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("patient_id_ext"),
        sa.CheckConstraint("age >= 0 AND age <= 150", name="chk_patients_age"),
        sa.CheckConstraint(
            "gender IN ('M', 'F')", name="chk_patients_gender"
        ),
    )
    op.create_index(
        "idx_patients_external_id", "patients", ["patient_id_ext"], unique=False
    )
    op.create_index("idx_patients_user_id", "patients", ["user_id"], unique=False)

    # ============================================================
    # Create images table
    # ============================================================
    op.create_table(
        "images",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "patient_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("patients.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("image_index", sa.String(50), nullable=False),
        sa.Column("file_path", sa.String(500), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False, default=1024),
        sa.Column("height", sa.Integer(), nullable=False, default=1024),
        sa.Column("view_position", sa.String(10), nullable=True),
        sa.Column("original_dcm_width", sa.Integer(), nullable=True),
        sa.Column("original_dcm_height", sa.Integer(), nullable=True),
        sa.Column("pixel_spacing_x", sa.Float(), nullable=True),
        sa.Column("pixel_spacing_y", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("image_index"),
        sa.CheckConstraint(
            "view_position IN ('AP', 'PA', 'UNKNOWN') OR view_position IS NULL",
            name="chk_images_view_position",
        ),
    )
    op.create_index("idx_images_patient_id", "images", ["patient_id"], unique=False)
    op.create_index(
        "idx_images_image_index", "images", ["image_index"], unique=False
    )
    op.create_index(
        "idx_images_view_position", "images", ["view_position"], unique=False
    )
    op.create_index(
        "idx_images_patient_created",
        "images",
        ["patient_id", "created_at"],
        unique=False,
    )

    # ============================================================
    # Create image_labels table
    # ============================================================
    op.create_table(
        "image_labels",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "image_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("images.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("label_name", sa.String(50), nullable=False),
        sa.Column("is_nlp_mined", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "image_id", "label_name", name="uq_image_labels_image_label"
        ),
        sa.CheckConstraint(
            f"label_name IN ({LABEL_LIST})",
            name="chk_image_labels_valid_label",
        ),
    )
    op.create_index(
        "idx_image_labels_image_id", "image_labels", ["image_id"], unique=False
    )
    op.create_index(
        "idx_image_labels_label_name",
        "image_labels",
        ["label_name"],
        unique=False,
    )

    # ============================================================
    # Create predictions table
    # ============================================================
    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "image_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("images.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("label_name", sa.String(50), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("predicted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("is_approved", sa.Boolean(), nullable=False, default=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="chk_predictions_confidence",
        ),
        sa.CheckConstraint(
            f"label_name IN ({LABEL_LIST})",
            name="chk_predictions_valid_label",
        ),
    )
    op.create_index(
        "idx_predictions_image_id", "predictions", ["image_id"], unique=False
    )
    op.create_index(
        "idx_predictions_model_version",
        "predictions",
        ["model_version"],
        unique=False,
    )
    op.create_index(
        "idx_predictions_created_by",
        "predictions",
        ["created_by"],
        unique=False,
    )
    op.create_index(
        "idx_predictions_label_conf",
        "predictions",
        ["label_name", "confidence"],
        unique=False,
    )
    op.create_index(
        "idx_predictions_predicted_at",
        "predictions",
        ["predicted_at"],
        unique=False,
    )
    op.create_index(
        "idx_predictions_image_predicted",
        "predictions",
        ["image_id", "predicted_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_table("predictions")
    op.drop_table("image_labels")
    op.drop_table("images")
    op.drop_table("patients")
    op.drop_table("users")