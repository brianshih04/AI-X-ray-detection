"""CRUD operations for Prediction model."""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.models import Image, Prediction
from src.schemas.prediction import PredictionCreate, PredictionUpdate

from .crud_utils import CRUDBase


class PredictionCRUD(CRUDBase[Prediction, PredictionCreate, PredictionUpdate]):
    """CRUD operations for Prediction model."""

    async def get_with_image(
        self,
        db: AsyncSession,
        obj_id: uuid.UUID,
    ) -> Optional[Prediction]:
        """Get prediction with image relationship."""
        result = await db.execute(
            select(Prediction)
            .options(selectinload(Prediction.image))
            .where(Prediction.id == obj_id)
        )
        return result.scalar_one_or_none()

    async def get_by_image(
        self,
        db: AsyncSession,
        image_id: uuid.UUID,
        *,
        model_version: Optional[str] = None,
        threshold: float = 0.0,
    ) -> list[Prediction]:
        """Get predictions for an image, optionally filtered."""
        query = select(Prediction).where(Prediction.image_id == image_id)

        if model_version:
            query = query.where(Prediction.model_version == model_version)
        if threshold > 0:
            query = query.where(Prediction.confidence >= threshold)

        query = query.order_by(Prediction.confidence.desc())
        result = await db.execute(query)
        return list(result.scalars().all())

    async def get_latest_by_image(
        self,
        db: AsyncSession,
        image_id: uuid.UUID,
    ) -> list[Prediction]:
        """Get latest predictions (most recent predicted_at) per label."""
        # Subquery for latest prediction per label
        latest_subq = (
            select(
                Prediction.label_name,
                func.max(Prediction.predicted_at).label("max_predicted"),
            )
            .where(Prediction.image_id == image_id)
            .group_by(Prediction.label_name)
            .subquery()
        )

        result = await db.execute(
            select(Prediction)
            .join(
                latest_subq,
                (Prediction.label_name == latest_subq.c.label_name)
                & (Prediction.predicted_at == latest_subq.c.max_predicted),
            )
            .where(Prediction.image_id == image_id)
        )
        return list(result.scalars().all())

    async def get_multi_by_model_version(
        self,
        db: AsyncSession,
        model_version: str,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[Prediction]:
        """Get predictions by model version."""
        result = await db.execute(
            select(Prediction)
            .where(Prediction.model_version == model_version)
            .order_by(Prediction.predicted_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def approve(
        self,
        db: AsyncSession,
        obj_id: uuid.UUID,
        notes: Optional[str] = None,
    ) -> Optional[Prediction]:
        """Approve a prediction with optional notes."""
        prediction = await db.get(Prediction, obj_id)
        if prediction:
            prediction.is_approved = True
            if notes:
                prediction.notes = notes
            await db.flush()
            await db.refresh(prediction)
        return prediction

    async def create_batch(
        self,
        db: AsyncSession,
        predictions: list[PredictionCreate],
    ) -> list[Prediction]:
        """Bulk create predictions."""
        db_preds = [Prediction(**pred.model_dump()) for pred in predictions]
        db.add_all(db_preds)
        await db.flush()
        return db_preds

    async def count_by_label(
        self,
        db: AsyncSession,
        model_version: Optional[str] = None,
        min_confidence: float = 0.0,
    ) -> dict[str, int]:
        """Count predictions per label."""
        query = select(
            Prediction.label_name,
            func.count(Prediction.id).label("count"),
        ).group_by(Prediction.label_name)

        if model_version:
            query = query.where(Prediction.model_version == model_version)
        if min_confidence > 0:
            query = query.where(Prediction.confidence >= min_confidence)

        result = await db.execute(query)
        return {row.label_name: row.count for row in result.all()}


prediction_crud = PredictionCRUD(Prediction)