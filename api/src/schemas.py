"""Pydantic schemas for API request/response contracts."""
from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ─── Auth ────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="seconds until expiry")


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(min_length=6, max_length=100)
    full_name: Optional[str] = None


class UserOut(BaseModel):
    id: UUID
    username: str
    email: str
    full_name: Optional[str] = None
    is_active: bool
    is_admin: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Patient ─────────────────────────────────────────

class PatientOut(BaseModel):
    id: UUID
    patient_id: str
    age: Optional[int] = None
    sex: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class PatientListResponse(BaseModel):
    patients: list[PatientOut]
    total: int
    page: int
    page_size: int


# ─── Image ───────────────────────────────────────────

class ImageOut(BaseModel):
    id: UUID
    patient_id: UUID
    file_path: str
    view_position: Optional[str] = None
    original_size_kb: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ImageLabelOut(BaseModel):
    id: UUID
    label: str
    value: int
    source: str = "nih"

    model_config = {"from_attributes": True}


class ImageDetailOut(ImageOut):
    labels: list[ImageLabelOut] = []


class ImageListResponse(BaseModel):
    images: list[ImageOut]
    total: int
    page: int
    page_size: int


# ─── Prediction ──────────────────────────────────────

class PredictionResult(BaseModel):
    label: str
    confidence: float


class PredictRequest(BaseModel):
    """Client sends image bytes as multipart file; this is for documentation."""
    pass


class PredictResponse(BaseModel):
    id: UUID
    image_id: Optional[UUID] = None
    model_version: str
    results: list[PredictionResult]
    top_prediction: str
    top_confidence: float
    processing_time_ms: float
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionDetailOut(PredictResponse):
    user_id: Optional[UUID] = None


# ─── Model Info ──────────────────────────────────────

class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str  # onnx / torchscript
    labels: list[str]
    input_shape: list[int]
    device: str
    batch_size: int
    confidence_threshold: float


# ─── Generic ─────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int = 400
