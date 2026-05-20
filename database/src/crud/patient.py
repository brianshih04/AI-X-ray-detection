"""CRUD operations for Patient model."""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models import Image, Patient
from src.schemas.patient import PatientCreate, PatientUpdate

from .crud_utils import CRUDBase


class PatientCRUD(CRUDBase[Patient, PatientCreate, PatientUpdate]):
    """CRUD operations for Patient model."""

    async def get_by_external_id(
        self,
        db: AsyncSession,
        patient_id_ext: str,
    ) -> Optional[Patient]:
        """Get patient by external NIH Patient ID."""
        result = await db.execute(
            select(Patient).where(Patient.patient_id_ext == patient_id_ext)
        )
        return result.scalar_one_or_none()

    async def get_with_image_count(
        self,
        db: AsyncSession,
        obj_id: uuid.UUID,
    ) -> Optional[Patient]:
        """Get patient with image count."""
        result = await db.execute(
            select(Patient)
            .options(selectinload(Patient.images))
            .where(Patient.id == obj_id)
        )
        return result.scalar_one_or_none()

    async def get_multi_with_counts(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        user_id: Optional[uuid.UUID] = None,
    ) -> list[tuple[Patient, int]]:
        """Get patients with image counts."""
        # Subquery for image count per patient
        img_count_subq = (
            select(Image.patient_id, func.count(Image.id).label("image_count"))
            .group_by(Image.patient_id)
            .subquery()
        )

        query = (
            select(Patient, img_count_subq.c.image_count)
            .outerjoin(img_count_subq, Patient.id == img_count_subq.c.patient_id)
        )

        if user_id is not None:
            query = query.where(Patient.user_id == user_id)

        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.all())

    async def search_by_external_id(
        self,
        db: AsyncSession,
        query: str,
        *,
        limit: int = 20,
    ) -> list[Patient]:
        """Search patients by external ID prefix."""
        result = await db.execute(
            select(Patient)
            .where(Patient.patient_id_ext.ilike(f"{query}%"))
            .limit(limit)
        )
        return list(result.scalars().all())


patient_crud = PatientCRUD(Patient)