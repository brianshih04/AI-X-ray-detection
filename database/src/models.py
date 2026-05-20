"""
SQLAlchemy ORM models for ChestXpert database.

Five core entities:
    - User      : authentication & authorization
    - Patient   : patient records linked to NIH metadata
    - Image     : X-ray image records
    - ImageLabel: NLP-mined disease labels (from NIH reports)
    - Prediction: model prediction results

Design goals:
    - UUID primary keys for distributed safety
    - Audit timestamps on all entities
    - Strict constraints at DB level
    - Indexes for T7 API query patterns
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from database import Base

if TYPE_CHECKING:
    pass


# ============================================================
# Constants
# ============================================================

NIH_LABELS: list[str] = [
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

LABEL_SET: set[str] = set(NIH_LABELS)


# ============================================================
# User Model
# ============================================================

class User(Base):
    """
    Application user (physician, radiologist, admin).

    Supports both regular users and superusers for admin operations.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Relationships
    patients: Mapped[list["Patient"]] = relationship(
        "Patient",
        back_populates="assigned_user",
        lazy="noload",
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction",
        back_populates="creator",
        lazy="noload",
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username={self.username})>"


# ============================================================
# Patient Model
# ============================================================

class Patient(Base):
    """
    Patient record.

    Maps to NIH PatientID in the source CSV metadata.
    """

    __tablename__ = "patients"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    patient_id_ext: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )
    age: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    gender: Mapped[str] = mapped_column(
        String(1),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False,
    )

    # Constraints
    __table_args__ = (
        CheckConstraint("age >= 0 AND age <= 150", name="chk_patients_age"),
        CheckConstraint(
            "gender IN ('M', 'F')",
            name="chk_patients_gender",
        ),
    )

    # Relationships
    assigned_user: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="patients",
    )
    images: Mapped[list["Image"]] = relationship(
        "Image",
        back_populates="patient",
        lazy="noload",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Patient(id={self.id}, ext={self.patient_id_ext})>"


# ============================================================
# Image Model
# ============================================================

class Image(Base):
    """
    Chest X-ray image record.

    Represents a single image from the NIH ChestX-ray14 dataset.
    Metadata from DICOM headers and CSV is preserved.
    """

    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    patient_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patients.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    image_index: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )
    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )
    width: Mapped[int] = mapped_column(
        Integer,
        default=1024,
        nullable=False,
    )
    height: Mapped[int] = mapped_column(
        Integer,
        default=1024,
        nullable=False,
    )
    view_position: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        index=True,
    )
    original_dcm_width: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    original_dcm_height: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    pixel_spacing_x: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    pixel_spacing_y: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "view_position IN ('AP', 'PA', 'UNKNOWN') OR view_position IS NULL",
            name="chk_images_view_position",
        ),
        Index("idx_images_patient_created", "patient_id", "created_at"),
    )

    # Relationships
    patient: Mapped["Patient"] = relationship(
        "Patient",
        back_populates="images",
    )
    labels: Mapped[list["ImageLabel"]] = relationship(
        "ImageLabel",
        back_populates="image",
        lazy="noload",
        cascade="all, delete-orphan",
    )
    predictions: Mapped[list["Prediction"]] = relationship(
        "Prediction",
        back_populates="image",
        lazy="noload",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Image(id={self.id}, index={self.image_index})>"


# ============================================================
# ImageLabel Model (NLP-mined from NIH reports)
# ============================================================

class ImageLabel(Base):
    """
    Disease label mined from NIH radiology reports via NLP.

    These labels are NOT human-annotated — they come from
    SNOMED-CT / UMLS text mining on free-text reports.
    Use with caution (see NIH analysis report).
    """

    __tablename__ = "image_labels"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    label_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    is_nlp_mined: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )

    # Constraints
    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "label_name",
            name="uq_image_labels_image_label",
        ),
        CheckConstraint(
            "label_name IN ("
            "'Atelectasis','Cardiomegaly','Effusion','Infiltration',"
            "'Mass','Nodule','Pneumonia','Pneumothorax',"
            "'Emphysema','Fibrosis','Pleural_Thickening',"
            "'Consolidation','Edema','Hernia'"
            ")",
            name="chk_image_labels_valid_label",
        ),
    )

    # Relationships
    image: Mapped["Image"] = relationship(
        "Image",
        back_populates="labels",
    )

    def __repr__(self) -> str:
        return f"<ImageLabel(id={self.id}, label={self.label_name})>"


# ============================================================
# Prediction Model
# ============================================================

class Prediction(Base):
    """
    Model prediction result.

    Stores per-label confidence scores from the chest X-ray classifier.
    Supports multiple model versions for A/B testing and versioning.
    """

    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    model_version: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    label_name: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    predicted_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    created_by: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    is_approved: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
    )
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
    )

    # Constraints
    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="chk_predictions_confidence",
        ),
        CheckConstraint(
            "label_name IN ("
            "'Atelectasis','Cardiomegaly','Effusion','Infiltration',"
            "'Mass','Nodule','Pneumonia','Pneumothorax',"
            "'Emphysema','Fibrosis','Pleural_Thickening',"
            "'Consolidation','Edema','Hernia'"
            ")",
            name="chk_predictions_valid_label",
        ),
        Index("idx_predictions_label_conf", "label_name", "confidence"),
        Index("idx_predictions_image_predicted", "image_id", "predicted_at"),
    )

    # Relationships
    image: Mapped["Image"] = relationship(
        "Image",
        back_populates="predictions",
    )
    creator: Mapped[Optional["User"]] = relationship(
        "User",
        back_populates="predictions",
    )

    def __repr__(self) -> str:
        return (
            f"<Prediction(id={self.id}, "
            f"label={self.label_name}, "
            f"conf={self.confidence:.3f})>"
        )
