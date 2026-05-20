"""SQLAlchemy ORM models — mirrors T1 schema (users, patients, images, image_labels, predictions)."""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, DateTime, ForeignKey, Float, Integer, Boolean,
    Text, CheckConstraint, Index, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from src.database import Base


def _utcnow():
    return datetime.utcnow()


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    predictions = relationship("Prediction", back_populates="user")


class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(String(50), unique=True, nullable=False, index=True)
    age = Column(Integer, CheckConstraint("age >= 0 AND age <= 150"))
    sex = Column(String(10), CheckConstraint("sex IN ('M', 'F', 'Unknown')"))
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    images = relationship("Image", back_populates="patient")


class Image(Base):
    __tablename__ = "images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    file_path = Column(String(500), nullable=False)
    view_position = Column(String(20))
    original_size_kb = Column(Integer)
    width = Column(Integer)
    height = Column(Integer)
    created_at = Column(DateTime(timezone=True), default=_utcnow)

    patient = relationship("Patient", back_populates="images")
    labels = relationship("ImageLabel", back_populates="image", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="image", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_images_patient_id", "patient_id"),
    )


class ImageLabel(Base):
    __tablename__ = "image_labels"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    label = Column(String(50), nullable=False)
    value = Column(Integer, CheckConstraint("value IN (0, 1)"), nullable=False)
    source = Column(String(20), default="nih")  # nih, manual

    image = relationship("Image", back_populates="labels")

    __table_args__ = (
        Index("ix_image_labels_image_id", "image_id"),
        Index("ix_image_labels_label", "label"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    image_id = Column(UUID(as_uuid=True), ForeignKey("images.id", ondelete="SET NULL"), nullable=True)
    model_version = Column(String(50), nullable=False)
    confidence_scores = Column(Text)  # JSON string of {label: score}
    top_prediction = Column(String(50))
    top_confidence = Column(Float, CheckConstraint("top_confidence >= 0 AND top_confidence <= 1"))
    processing_time_ms = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default="NOW()")

    user = relationship("User", back_populates="predictions")
    image = relationship("Image", back_populates="predictions")
