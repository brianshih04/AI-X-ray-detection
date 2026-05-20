"""CRUD operations for all models."""
import json
from typing import Optional
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models import User, Patient, Image, ImageLabel, Prediction


# ─── User ────────────────────────────────────────────

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: UUID) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, username: str, email: str, hashed_password: str, full_name: Optional[str] = None) -> User:
    user = User(username=username, email=email, hashed_password=hashed_password, full_name=full_name)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ─── Patient ─────────────────────────────────────────

def list_patients(db: Session, page: int = 1, page_size: int = 20) -> tuple[list[Patient], int]:
    total = db.query(func.count(Patient.id)).scalar() or 0
    offset = (page - 1) * page_size
    patients = db.query(Patient).order_by(Patient.created_at.desc()).offset(offset).limit(page_size).all()
    return patients, total


def get_patient_by_id(db: Session, patient_id: UUID) -> Optional[Patient]:
    return db.query(Patient).filter(Patient.id == patient_id).first()


# ─── Image ───────────────────────────────────────────

def list_images_by_patient(db: Session, patient_id: UUID, page: int = 1, page_size: int = 20) -> tuple[list[Image], int]:
    total = db.query(func.count(Image.id)).filter(Image.patient_id == patient_id).scalar() or 0
    offset = (page - 1) * page_size
    images = (
        db.query(Image)
        .filter(Image.patient_id == patient_id)
        .order_by(Image.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return images, total


def get_image_by_id(db: Session, image_id: UUID) -> Optional[Image]:
    return db.query(Image).filter(Image.id == image_id).first()


def get_image_labels(db: Session, image_id: UUID) -> list[ImageLabel]:
    return db.query(ImageLabel).filter(ImageLabel.image_id == image_id).all()


# ─── Prediction ──────────────────────────────────────

def get_prediction_by_id(db: Session, prediction_id: UUID) -> Optional[Prediction]:
    return db.query(Prediction).filter(Prediction.id == prediction_id).first()


def create_prediction(
    db: Session,
    user_id: Optional[UUID],
    image_id: Optional[UUID],
    model_version: str,
    results: dict[str, float],
    processing_time_ms: float,
) -> Prediction:
    # Find top prediction
    top_label = max(results, key=results.get) if results else None
    top_confidence = results.get(top_label, 0.0) if top_label else 0.0

    prediction = Prediction(
        user_id=user_id,
        image_id=image_id,
        model_version=model_version,
        confidence_scores=json.dumps(results),
        top_prediction=top_label,
        top_confidence=top_confidence,
        processing_time_ms=processing_time_ms,
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction
