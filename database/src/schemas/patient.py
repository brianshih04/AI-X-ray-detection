"""Pydantic schemas for Patient model."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# Patient Schemas
# ============================================================

class PatientBase(BaseModel):
    """Base patient fields."""

    patient_id_ext: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="External NIH Patient ID",
    )
    age: int = Field(..., ge=0, le=150, description="Patient age")
    gender: str = Field(..., pattern="^(M|F)$", description="Gender: M or F")
    user_id: Optional[uuid.UUID] = Field(
        None,
        description="Assigned physician user ID",
    )


class PatientCreate(PatientBase):
    """Schema for creating a patient."""

    pass


class PatientUpdate(BaseModel):
    """Schema for updating patient fields."""

    age: Optional[int] = Field(None, ge=0, le=150)
    gender: Optional[str] = Field(None, pattern="^(M|F)$")
    user_id: Optional[uuid.UUID] = None


class PatientInDB(PatientBase):
    """Full patient record from database."""

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PatientBrief(BaseModel):
    """Brief patient info for listings."""

    id: uuid.UUID
    patient_id_ext: str
    age: int
    gender: str

    model_config = ConfigDict(from_attributes=True)


class PatientWithImages(PatientInDB):
    """Patient with associated image count."""

    image_count: int = Field(default=0, description="Number of images")

    model_config = ConfigDict(from_attributes=True)