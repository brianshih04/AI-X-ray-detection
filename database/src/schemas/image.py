"""Pydantic schemas for Image and ImageLabel models."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from .patient import PatientBrief


# ============================================================
# ImageLabel Schemas
# ============================================================

class ImageLabelBase(BaseModel):
    """Base image label fields."""

    label_name: str = Field(..., description="NIH disease label name")
    is_nlp_mined: bool = Field(
        default=True,
        description="True if label is from NLP text mining",
    )


class ImageLabelCreate(ImageLabelBase):
    """Schema for creating an image label."""

    image_id: uuid.UUID


class ImageLabelInDB(ImageLabelBase):
    """Image label from database."""

    id: uuid.UUID
    image_id: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================
# Image Schemas
# ============================================================

class ImageBase(BaseModel):
    """Base image fields."""

    image_index: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="NIH image filename (e.g. 00000001_000.png)",
    )
    file_path: str = Field(
        ...,
        max_length=500,
        description="Local filesystem path to image",
    )
    width: int = Field(default=1024, ge=1, description="Image width in pixels")
    height: int = Field(default=1024, ge=1, description="Image height in pixels")
    view_position: Optional[str] = Field(
        None,
        pattern="^(AP|PA|UNKNOWN)$",
        description="X-ray view position",
    )
    original_dcm_width: Optional[int] = Field(None, ge=1)
    original_dcm_height: Optional[int] = Field(None, ge=1)
    pixel_spacing_x: Optional[float] = Field(None, ge=0)
    pixel_spacing_y: Optional[float] = Field(None, ge=0)


class ImageCreate(ImageBase):
    """Schema for creating an image record."""

    patient_id: uuid.UUID


class ImageUpdate(BaseModel):
    """Schema for updating image fields."""

    file_path: Optional[str] = Field(None, max_length=500)
    view_position: Optional[str] = Field(None, pattern="^(AP|PA|UNKNOWN)$")
    pixel_spacing_x: Optional[float] = Field(None, ge=0)
    pixel_spacing_y: Optional[float] = Field(None, ge=0)


class ImageInDB(ImageBase):
    """Full image record from database."""

    id: uuid.UUID
    patient_id: uuid.UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ImageBrief(BaseModel):
    """Brief image info for listings."""

    id: uuid.UUID
    image_index: str
    file_path: str
    width: int
    height: int
    view_position: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ImageWithLabels(ImageInDB):
    """Image with its NLP-mined labels."""

    labels: list[ImageLabelInDB] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class ImageWithPatient(ImageInDB):
    """Image with patient info."""

    patient: PatientBrief

    model_config = ConfigDict(from_attributes=True)


class ImageDetail(ImageInDB):
    """Detailed image with patient and labels."""

    patient: PatientBrief
    labels: list[ImageLabelInDB] = Field(default_factory=list)
    prediction_count: int = Field(default=0, description="Number of predictions")

    model_config = ConfigDict(from_attributes=True)