"""Pydantic schemas for Prediction model."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .image import ImageBrief


# ============================================================
# Prediction Schemas
# ============================================================

class PredictionBase(BaseModel):
    """Base prediction fields."""

    model_version: str = Field(
        ...,
        max_length=50,
        description="Model version identifier",
    )
    label_name: str = Field(..., description="Predicted disease label")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score",
    )


class PredictionCreate(PredictionBase):
    """Schema for creating a prediction."""

    image_id: uuid.UUID
    predicted_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[uuid.UUID] = None


class PredictionUpdate(BaseModel):
    """Schema for updating prediction (approval, notes)."""

    is_approved: Optional[bool] = None
    notes: Optional[str] = None


class PredictionInDB(PredictionBase):
    """Prediction from database."""

    id: uuid.UUID
    image_id: uuid.UUID
    predicted_at: datetime
    created_by: Optional[uuid.UUID] = None
    is_approved: bool
    notes: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class PredictionWithImage(PredictionInDB):
    """Prediction with associated image."""

    image: ImageBrief

    model_config = ConfigDict(from_attributes=True)


# ============================================================
# Batch Prediction Schemas
# ============================================================

class PredictionResult(BaseModel):
    """Single label prediction result."""

    label_name: str
    confidence: float


class PredictionResponse(BaseModel):
    """API response for prediction request."""

    image_id: uuid.UUID
    image_index: str
    model_version: str
    predicted_at: datetime
    predictions: list[PredictionResult] = Field(
        default_factory=list,
        description="List of label predictions sorted by confidence",
    )


class PredictionBatchRequest(BaseModel):
    """Batch prediction request."""

    image_ids: list[uuid.UUID] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of image IDs to predict",
    )
    model_version: Optional[str] = Field(
        None,
        description="Override default model version",
    )
    threshold: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence threshold filter",
    )


class PredictionBatchResponse(BaseModel):
    """Batch prediction response."""

    results: list[PredictionResponse]
    total: int
    threshold_used: float
    model_version: str


# ============================================================
# Prediction History / Analytics
# ============================================================

class PredictionSummary(BaseModel):
    """Summary of predictions for an image."""

    total_predictions: int
    approved_count: int
    latest_model_version: Optional[str] = None
    latest_predicted_at: Optional[datetime] = None


class PredictionStats(BaseModel):
    """Statistics for a prediction set."""

    total: int
    by_label: dict[str, int] = Field(
        default_factory=dict,
        description="Count per label",
    )
    avg_confidence: float
    model_versions: list[str] = Field(default_factory=list)