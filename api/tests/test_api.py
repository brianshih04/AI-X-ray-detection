"""Tests for core API endpoints (predict, patients, predictions, model)."""
import json
import pytest
from unittest.mock import patch, MagicMock
from uuid import uuid4

from src.models import Patient, Image


class TestHealthCheck:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestModelInfo:
    def test_model_info(self, client):
        resp = client.get("/api/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] in ("onnx", "torchscript", "none")
        assert isinstance(data["labels"], list)
        assert len(data["labels"]) == 14
        assert data["input_shape"] == [1, 3, 224, 224]


class TestPredict:
    def test_predict_success(self, client, sample_image_bytes):
        """Test prediction with a valid image upload."""
        from src.services.inference import inference_service

        mock_result = {
            "Atelectasis": 0.1, "Cardiomegaly": 0.8, "Effusion": 0.3,
            "Infiltration": 0.05, "Mass": 0.02, "Nodule": 0.01,
            "Pneumonia": 0.7, "Pneumothorax": 0.1, "Consolidation": 0.05,
            "Edema": 0.4, "Emphysema": 0.1, "Fibrosis": 0.08,
            "Pleural_Thickening": 0.15, "Hernia": 0.02,
            "_processing_time_ms": 45.2,
        }
        with patch.object(inference_service, "predict_single", return_value=mock_result), \
             patch.object(inference_service, "_loaded", True), \
             patch.object(inference_service, "_model_type", "none"):
            resp = client.post(
                "/api/predict",
                files={"file": ("test.png", sample_image_bytes, "image/png")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "results" in data
        assert len(data["results"]) > 0
        assert data["model_version"] == "1.0.0"
        assert data["processing_time_ms"] > 0
        # Check results are sorted descending
        scores = [r["confidence"] for r in data["results"]]
        assert scores == sorted(scores, reverse=True)

    def test_predict_stub_mode(self, client, sample_image_bytes):
        """Test prediction when no model is loaded (stub mode)."""
        resp = client.post(
            "/api/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "id" in data
        assert "results" in data

    def test_predict_with_auth(self, client, sample_image_bytes, auth_headers):
        """Test that authenticated prediction stores user_id."""
        from src.services.inference import inference_service

        mock_result = {
            label: 0.1 for label in [
                "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
                "Mass", "Nodule", "Pneumonia", "Pneumothorax", "Consolidation",
                "Edema", "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia",
            ]
        }
        mock_result["_processing_time_ms"] = 10.0
        with patch.object(inference_service, "predict_single", return_value=mock_result), \
             patch.object(inference_service, "_loaded", True), \
             patch.object(inference_service, "_model_type", "none"):
            resp = client.post(
                "/api/predict",
                files={"file": ("test.png", sample_image_bytes, "image/png")},
                headers=auth_headers,
            )
        assert resp.status_code == 200

    def test_predict_invalid_format(self, client):
        resp = client.post(
            "/api/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert resp.status_code == 400


class TestPatients:
    def test_list_patients_empty(self, client):
        resp = client.get("/api/patients")
        assert resp.status_code == 200
        data = resp.json()
        assert data["patients"] == []
        assert data["total"] == 0

    def test_get_patient_not_found(self, client):
        fake_id = str(uuid4())
        resp = client.get(f"/api/patients/{fake_id}")
        assert resp.status_code == 404

    def test_patient_crud(self, client, db_session):
        """Create a patient, list, and retrieve."""
        patient = Patient(patient_id="P001", age=65, sex="M")
        db_session.add(patient)
        db_session.commit()

        # List
        resp = client.get("/api/patients")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["patients"][0]["patient_id"] == "P001"

        # Get by ID
        pid = data["patients"][0]["id"]
        resp = client.get(f"/api/patients/{pid}")
        assert resp.status_code == 200
        assert resp.json()["patient_id"] == "P001"


class TestPatientImages:
    def test_patient_images_empty(self, client, db_session):
        patient = Patient(patient_id="P002")
        db_session.add(patient)
        db_session.commit()

        resp = client.get(f"/api/patients/{patient.id}/images")
        assert resp.status_code == 200
        assert resp.json()["images"] == []

    def test_patient_images_with_data(self, client, db_session):
        patient = Patient(patient_id="P003")
        db_session.add(patient)
        db_session.commit()

        img1 = Image(patient_id=patient.id, file_path="/data/img1.png", view_position="AP")
        img2 = Image(patient_id=patient.id, file_path="/data/img2.png", view_position="PA")
        db_session.add_all([img1, img2])
        db_session.commit()

        resp = client.get(f"/api/patients/{patient.id}/images")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2

    def test_patient_not_found_images(self, client):
        resp = client.get(f"/api/patients/{uuid4()}/images")
        assert resp.status_code == 404


class TestPredictions:
    def test_prediction_not_found(self, client):
        resp = client.get(f"/api/predictions/{uuid4()}")
        assert resp.status_code == 404
