"""Pydantic schemas for User model."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ============================================================
# User Schemas
# ============================================================

class UserBase(BaseModel):
    """Base user fields (shared create/update)."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Login username",
    )
    email: EmailStr = Field(..., description="Email address")


class UserCreate(UserBase):
    """Schema for creating a new user."""

    password: str = Field(
        ...,
        min_length=8,
        max_length=100,
        description="Plain text password (hashed on creation)",
    )


class UserUpdate(BaseModel):
    """Schema for updating user fields (all optional)."""

    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=50,
    )
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


class UserInDB(UserBase):
    """Schema for user read from database."""

    id: uuid.UUID
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserPublic(UserBase):
    """Public user info (no sensitive fields)."""

    id: uuid.UUID
    is_active: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================
# Auth Schemas
# ============================================================

class Token(BaseModel):
    """JWT access token response."""

    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Data encoded in JWT token."""

    user_id: Optional[str] = None
    username: Optional[str] = None


class LoginRequest(BaseModel):
    """Login credentials."""

    username: str
    password: str