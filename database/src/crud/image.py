"""CRUD operations for Image and ImageLabel models."""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models import Image, ImageLabel, Patient
from src.schemas.image import ImageCreate, ImageLabelCreate

from .crud_utils import CRUDBase


# ============================================================
# Image CRUD
# ============================================================

class ImageCRUD(CRUDBase[Image, ImageCreate, dict]):
    """CRUD operations for Image model."""

    async def get_by_image_index(
        self,
        db: AsyncSession,
        image_index: str,
    ) -> Optional[Image]:
        """Get image by NIH image filename."""
        result = await db.execute(
            select(Image).where(Image.image_index == image_index)
        )
        return result.scalar_one_or_none()

    async def get_with_patient(
        self,
        db: AsyncSession,
        obj_id: uuid.UUID,
    ) -> Optional[Image]:
        """Get image with patient relationship."""
        result = await db.execute(
            select(Image)
            .options(selectinload(Image.patient))
            .where(Image.id == obj_id)
        )
        return result.scalar_one_or_none()

    async def get_with_labels(
        self,
        db: AsyncSession,
        obj_id: uuid.UUID,
    ) -> Optional[Image]:
        """Get image with NLP labels."""
        result = await db.execute(
            select(Image)
            .options(selectinload(Image.labels))
            .where(Image.id == obj_id)
        )
        return result.scalar_one_or_none()

    async def get_multi_by_patient(
        self,
        db: AsyncSession,
        patient_id: uuid.UUID,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Image]:
        """Get images for a specific patient."""
        result = await db.execute(
            select(Image)
            .where(Image.patient_id == patient_id)
            .order_by(Image.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_multi_with_pagination(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
        view_position: Optional[str] = None,
        patient_id: Optional[uuid.UUID] = None,
    ) -> list[Image]:
        """Get images with optional filters."""
        query = select(Image)

        if view_position:
            query = query.where(Image.view_position == view_position)
        if patient_id:
            query = query.where(Image.patient_id == patient_id)

        query = query.order_by(Image.created_at.desc()).offset(skip).limit(limit)
        result = await db.execute(query)
        return list(result.scalars().all())


# ============================================================
# ImageLabel CRUD
# ============================================================

class ImageLabelCRUD:
    """CRUD operations for ImageLabel model."""

    async def get_by_image(
        self,
        db: AsyncSession,
        image_id: uuid.UUID,
    ) -> list[ImageLabel]:
        """Get all labels for an image."""
        result = await db.execute(
            select(ImageLabel)
            .where(ImageLabel.image_id == image_id)
            .order_by(ImageLabel.label_name)
        )
        return list(result.scalars().all())

    async def create_many(
        self,
        db: AsyncSession,
        labels: list[ImageLabelCreate],
    ) -> list[ImageLabel]:
        """Bulk create image labels."""
        db_labels = [
            ImageLabel(**label.model_dump()) for label in labels
        ]
        db.add_all(db_labels)
        await db.flush()
        return db_labels

    async def delete_by_image(
        self,
        db: AsyncSession,
        image_id: uuid.UUID,
    ) -> int:
        """Delete all labels for an image, return count."""
        result = await db.execute(
            select(ImageLabel)
            .where(ImageLabel.image_id == image_id)
        )
        labels = result.scalars().all()
        count = len(labels)
        for label in labels:
            await db.delete(label)
        await db.flush()
        return count


image_crud = ImageCRUD(Image)
image_label_crud = ImageLabelCRUD()