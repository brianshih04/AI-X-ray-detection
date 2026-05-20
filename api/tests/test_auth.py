"""Tests for authentication endpoints."""
import pytest


class TestAuthRegister:
    def test_register_success(self, client):
        resp = client.post("/auth/register", json={
            "username": "newuser",
            "email": "new@example.com",
            "password": "securepass123",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "newuser"
        assert data["email"] == "new@example.com"
        assert "id" in data
        assert data["is_active"] is True

    def test_register_duplicate_username(self, client, db_session):
        # Register once
        client.post("/auth/register", json={
            "username": "dupuser",
            "email": "dup1@example.com",
            "password": "pass123",
        })
        # Duplicate
        resp = client.post("/auth/register", json={
            "username": "dupuser",
            "email": "dup2@example.com",
            "password": "pass123",
        })
        assert resp.status_code == 409

    def test_register_short_password(self, client):
        resp = client.post("/auth/register", json={
            "username": "short",
            "email": "short@example.com",
            "password": "abc",
        })
        assert resp.status_code == 422


class TestAuthLogin:
    def test_login_success(self, client):
        # Register first
        client.post("/auth/register", json={
            "username": "loginuser",
            "email": "login@example.com",
            "password": "mypass123",
        })
        # Login
        resp = client.post("/auth/login", json={
            "username": "loginuser",
            "password": "mypass123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_wrong_password(self, client):
        client.post("/auth/register", json={
            "username": "wronguser",
            "email": "wrong@example.com",
            "password": "correctpass",
        })
        resp = client.post("/auth/login", json={
            "username": "wronguser",
            "password": "wrongpass",
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post("/auth/login", json={
            "username": "ghost",
            "password": "nopass",
        })
        assert resp.status_code == 401


class TestAuthMe:
    def test_me_success(self, client, auth_headers):
        resp = client.get("/auth/me", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "testuser"
        assert data["email"] == "test@example.com"

    def test_me_invalid_token(self, client):
        resp = client.get("/auth/me", headers={"Authorization": "Bearer invalid.token.here"})
        assert resp.status_code == 401

    def test_me_no_token(self, client):
        resp = client.get("/auth/me")
        assert resp.status_code == 401
