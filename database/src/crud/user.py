"""CRUD operations for User model."""
from __future__ import annotations

import uuid
from typing import Optional

from passlib.context import CryptContext
from pydantic import EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models import User
from src.schemas.user import UserCreate, UserUpdate

from .crud_utils import CRUDBase

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserCRUD(CRUDBase[User, UserCreate, UserUpdate]):
    """CRUD operations for User model."""

    async def get_by_username(
        self,
        db: AsyncSession,
        username: str,
    ) -> Optional[User]:
        """Get user by username."""
        result = await db.execute(
            select(User).where(User.username == username)
        )
        return result.scalar_one_or_none()

    async def get_by_email(
        self,
        db: AsyncSession,
        email: str,
    ) -> Optional[User]:
        """Get user by email."""
        result = await db.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        db: AsyncSession,
        *,
        obj_in: UserCreate,
    ) -> User:
        """Create new user with hashed password."""
        db_obj = User(
            username=obj_in.username,
            email=obj_in.email,
            hashed_password=self.hash_password(obj_in.password),
        )
        db.add(db_obj)
        await db.flush()
        await db.refresh(db_obj)
        return db_obj

    async def update(
        self,
        db: AsyncSession,
        *,
        obj: User,
        obj_in: UserUpdate | dict,
    ) -> User:
        """Update user, handling password change."""
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)

        # Hash new password if provided
        if "password" in update_data and update_data["password"]:
            update_data["hashed_password"] = self.hash_password(
                update_data.pop("password")
            )
        elif "password" in update_data:
            update_data.pop("password")

        for field, value in update_data.items():
            if hasattr(obj, field):
                setattr(obj, field, value)

        await db.flush()
        await db.refresh(obj)
        return obj

    async def authenticate(
        self,
        db: AsyncSession,
        username: str,
        password: str,
    ) -> Optional[User]:
        """Authenticate user by username and password."""
        user = await self.get_by_username(db, username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        return user

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)


user_crud = UserCRUD(User)