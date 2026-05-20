"""CRUD utilities."""
from __future__ import annotations

import uuid
from typing import Any, Generic, TypeVar

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    """
    Base CRUD class with common operations.

    Type parameters:
        ModelType: SQLAlchemy model class
        CreateSchemaType: Pydantic create schema
        UpdateSchemaType: Pydantic update schema
    """

    def __init__(self, model: type[ModelType]):
        """Initialize with model class."""
        self.model = model

    async def get(
        self,
        db: AsyncSession,
        obj_id: uuid.UUID,
    ) -> ModelType | None:
        """Get by primary key."""
        return await db.get(self.model, obj_id)

    async def get_multi(
        self,
        db: AsyncSession,
        *,
        skip: int = 0,
        limit: int = 100,
    ) -> list[ModelType]:
        """Get multiple with pagination."""
        result = await db.execute(
            select(self.model)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_multi_by_ids(
        self,
        db: AsyncSession,
        ids: list[uuid.UUID],
    ) -> list[ModelType]:
        """Get multiple by list of IDs."""
        if not ids:
            return []
        result = await db.execute(
            select(self.model)
            .where(self.model.id.in_(ids))
        )
        return list(result.scalars().all())

    async def create(
        self,
        db: AsyncSession,
        *,
        obj_in: CreateSchemaType,
    ) -> ModelType:
        """Create new record."""
        obj = self.model(**obj_in.model_dump())
        db.add(obj)
        await db.flush()
        await db.refresh(obj)
        return obj

    async def update(
        self,
        db: AsyncSession,
        *,
        obj: ModelType,
        obj_in: UpdateSchemaType | dict[str, Any],
    ) -> ModelType:
        """Update existing record."""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)

        for field, value in update_data.items():
            if hasattr(obj, field):
                setattr(obj, field, value)

        await db.flush()
        await db.refresh(obj)
        return obj

    async def delete(
        self,
        db: AsyncSession,
        *,
        obj: ModelType,
    ) -> ModelType:
        """Delete record."""
        await db.delete(obj)
        await db.flush()
        return obj

    async def count(self, db: AsyncSession) -> int:
        """Count total records."""
        result = await db.execute(select(func.count()).select_from(self.model))
        return result.scalar_one()

    async def exists(
        self,
        db: AsyncSession,
        obj_id: uuid.UUID,
    ) -> bool:
        """Check if record exists."""
        result = await db.execute(
            select(func.count())
            .select_from(self.model)
            .where(self.model.id == obj_id)
        )
        return result.scalar_one() > 0


async def get_or_404(
    db: AsyncSession,
    model: type[Base],
    obj_id: uuid.UUID,
    error_message: str | None = None,
) -> Base:
    """Get by ID or raise HTTP 404."""
    obj = await db.get(model, obj_id)
    if obj is None:
        from fastapi import HTTPException, status
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_message or f"{model.__name__} not found",
        )
    return obj