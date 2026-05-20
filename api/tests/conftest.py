"""Conftest — shared fixtures for all tests."""
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import Base, get_db, init_engine
from src.main import app

# ─── In-memory SQLite for tests ──────────────────────

SQLALCHEMY_TEST_URL = "sqlite:///file::memory:?cache=shared&uri=true"

test_engine = create_engine(SQLALCHEMY_TEST_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    """Create all tables before each test, drop after."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def db_session():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def auth_token(client, db_session):
    """Register a test user and return JWT token."""
    response = client.post("/auth/register", json={
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpass123",
        "full_name": "Test User",
    })
    response = client.post("/auth/login", json={
        "username": "testuser",
        "password": "testpass123",
    })
    return response.json()["access_token"]


@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def sample_image_bytes():
    """Create a small 224x224 test PNG image."""
    img = Image.new("RGB", (224, 224), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def patch_model_loading():
    """Prevent actual model loading during tests (stub mode)."""
    from src.services.inference import InferenceService
    original = InferenceService.load_model
    InferenceService.load_model = lambda self: None
    yield
    InferenceService.load_model = original
