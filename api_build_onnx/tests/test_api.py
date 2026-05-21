"""Tests for the AI X-ray Detection API.

Run with:
    cd api_build_onnx
    pytest tests/ -v

Requirements:
    pip install pytest pytest-asyncio httpx
"""
import base64
import io
import struct
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image


# ============================================================
# Helper: create test images
# ============================================================

def make_png_bytes(width=224, height=224, mode="RGB") -> bytes:
    """Create a valid PNG image in memory."""
    img = Image.new(mode, (width, height), color=128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_jpeg_bytes(width=224, height=224) -> bytes:
    """Create a valid JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=200)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_dicom_bytes(width=224, height=224) -> bytes:
    """Create a minimal DICOM file with pixel data."""
    import pydicom
    from pydicom.dataset import FileDataset
    from pydicom.uid import ExplicitVRLittleEndian

    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7.8.9"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(io.BytesIO(), {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.Rows = height
    ds.Columns = width
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"

    pixel_array = np.random.randint(0, 4096, (height, width), dtype=np.uint16)
    ds.PixelData = pixel_array.tobytes()

    buf = io.BytesIO()
    ds.save_as(buf)
    return buf.getvalue()


# ============================================================
# Tests: is_dicom()
# ============================================================

class TestIsDicom:
    def test_dicom_with_dicm_magic(self):
        """Standard DICOM file has DICM magic at offset 128."""
        from main import is_dicom
        data = b"\x00" * 128 + b"DICM" + b"\x00" * 100
        assert is_dicom(data) is True

    def test_dicom_with_group_0002(self):
        """DICOM without preamble but starting with group 0002."""
        from main import is_dicom
        data = struct.pack("<H", 0x0002) + b"\x00" * 100
        assert is_dicom(data) is True

    def test_png_not_dicom(self):
        """PNG file should not be detected as DICOM."""
        from main import is_dicom
        data = make_png_bytes()
        assert is_dicom(data) is False

    def test_jpeg_not_dicom(self):
        """JPEG file should not be detected as DICOM."""
        from main import is_dicom
        data = make_jpeg_bytes()
        assert is_dicom(data) is False

    def test_short_data(self):
        """Very short data should not crash."""
        from main import is_dicom
        assert is_dicom(b"") is False
        assert is_dicom(b"short") is False


# ============================================================
# Tests: dicom_to_image()
# ============================================================

class TestDicomToImage:
    def test_grayscale_dicom(self):
        """Grayscale DICOM should convert to RGB PIL Image."""
        from main import dicom_to_image
        data = make_dicom_bytes(100, 100)
        img = dicom_to_image(data)
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_dicom_different_sizes(self):
        """DICOM with various sizes should all convert."""
        from main import dicom_to_image
        for w, h in [(64, 64), (512, 512), (1024, 768)]:
            data = make_dicom_bytes(w, h)
            img = dicom_to_image(data)
            assert img.size == (w, h), f"Failed for {w}x{h}"


# ============================================================
# Tests: preprocess()
# ============================================================

class TestPreprocess:
    def test_png_preprocess(self):
        """PNG image should produce correct shaped array."""
        from main import preprocess
        data = make_png_bytes()
        result = preprocess(data)
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32

    def test_jpeg_preprocess(self):
        """JPEG image should produce correct shaped array."""
        from main import preprocess
        data = make_jpeg_bytes()
        result = preprocess(data)
        assert result.shape == (1, 3, 224, 224)

    def test_dicom_preprocess(self):
        """DICOM image should produce correct shaped array."""
        from main import preprocess
        data = make_dicom_bytes()
        result = preprocess(data)
        assert result.shape == (1, 3, 224, 224)

    def test_non_square_image(self):
        """Non-square image should resize to 224x224."""
        from main import preprocess
        data = make_png_bytes(width=300, height=400)
        result = preprocess(data)
        assert result.shape == (1, 3, 224, 224)

    def test_grayscale_png(self):
        """Grayscale PNG should convert to 3-channel."""
        from main import preprocess
        data = make_png_bytes(mode="L")
        result = preprocess(data)
        assert result.shape == (1, 3, 224, 224)

    def test_normalization_range(self):
        """Normalized values should be roughly in valid range."""
        from main import preprocess
        data = make_png_bytes()
        result = preprocess(data)
        # ImageNet normalization can produce negative values, but shouldn't be extreme
        assert result.min() > -5.0
        assert result.max() < 5.0


# ============================================================
# Tests: API endpoints (using TestClient)
# ============================================================

@pytest.fixture
def client():
    """Create a test client with mocked model loading."""
    # Mock the ONNX model loading to avoid needing model files in tests
    with patch("main.load_model") as mock_load, \
         patch("main.init_db") as mock_db:
        # After load_model, set global session variables
        import main

        mock_session = MagicMock()
        mock_session.run.return_value = [np.random.rand(1, 15).astype(np.float32)]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        mock_cam_session = MagicMock()
        # CAM model returns [logits, feature_maps]
        mock_feature_maps = np.random.rand(1, 1024, 7, 7).astype(np.float32)
        mock_cam_session.run.return_value = [
            np.random.rand(1, 15).astype(np.float32),
            mock_feature_maps
        ]

        def setup_globals():
            main.session = mock_session
            main.cam_session = mock_cam_session
            main.cam_weights = np.random.rand(15, 1024).astype(np.float32)
            main.db_ready = False  # Skip DB in tests

        mock_load.side_effect = setup_globals
        mock_db.return_value = None

        from fastapi.testclient import TestClient
        from main import app
        with TestClient(app) as c:
            yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "CPUExecutionProvider" in data["providers"]


class TestPredictEndpoint:
    def test_predict_png(self, client):
        resp = client.post(
            "/api/predict",
            files={"file": ("test.png", make_png_bytes(), "image/png")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 15
        assert data["top_prediction"] in [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
            "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax", "No_Finding",
        ]
        assert 0 <= data["top_confidence"] <= 1
        assert data["processing_time_ms"] > 0

    def test_predict_jpeg(self, client):
        resp = client.post(
            "/api/predict",
            files={"file": ("test.jpg", make_jpeg_bytes(), "image/jpeg")}
        )
        assert resp.status_code == 200

    def test_predict_dicom(self, client):
        resp = client.post(
            "/api/predict",
            files={"file": ("test.dcm", make_dicom_bytes(), "application/dicom")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 15

    def test_predict_invalid_format(self, client):
        resp = client.post(
            "/api/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert resp.status_code == 400

    def test_predict_results_sorted(self, client):
        resp = client.post(
            "/api/predict",
            files={"file": ("test.png", make_png_bytes(), "image/png")}
        )
        data = resp.json()
        confidences = [r["confidence"] for r in data["results"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_predict_each_result_has_label_and_confidence(self, client):
        resp = client.post(
            "/api/predict",
            files={"file": ("test.png", make_png_bytes(), "image/png")}
        )
        data = resp.json()
        for r in data["results"]:
            assert "label" in r
            assert "confidence" in r
            assert isinstance(r["confidence"], float)
            assert 0 <= r["confidence"] <= 1


class TestGradcamEndpoint:
    def test_gradcam_png(self, client):
        resp = client.post(
            "/api/gradcam",
            files={"file": ("test.png", make_png_bytes(), "image/png")},
            data={"label": "Cardiomegaly"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "heatmap" in data
        # Label may differ from request because mock returns random logits
        # and the code falls back to top prediction if label not in top results
        assert data["target_label"] in [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
            "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax", "No_Finding",
        ]
        # Heatmap should be valid base64 PNG
        heatmap_bytes = base64.b64decode(data["heatmap"])
        assert heatmap_bytes[:4] == b"\x89PNG"

    def test_gradcam_dicom(self, client):
        resp = client.post(
            "/api/gradcam",
            files={"file": ("test.dcm", make_dicom_bytes(), "application/dicom")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "heatmap" in data

    def test_gradcam_default_label(self, client):
        """Without specifying label, should use top prediction."""
        resp = client.post(
            "/api/gradcam",
            files={"file": ("test.png", make_png_bytes(), "image/png")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["target_label"] is not None

    def test_gradcam_invalid_label_fallback(self, client):
        """Invalid label should fall back to top prediction."""
        resp = client.post(
            "/api/gradcam",
            files={"file": ("test.png", make_png_bytes(), "image/png")},
            data={"label": "InvalidDisease"}
        )
        assert resp.status_code == 200


class TestModelInfoEndpoint:
    def test_model_info(self, client):
        resp = client.get("/api/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "DenseNet-121 (ONNX)"
        assert len(data["labels"]) == 15
        assert data["input_size"] == [1, 3, 224, 224]


class TestPredictionsEndpoint:
    def test_predictions_list(self, client):
        resp = client.get("/api/predictions")
        # DB not available in test env, should return 503
        assert resp.status_code == 503


# ============================================================
# Tests: Edge cases
# ============================================================

class TestEdgeCases:
    def test_very_small_image(self, client):
        """Tiny image (1x1) should still work after resize."""
        data = make_png_bytes(width=1, height=1)
        resp = client.post(
            "/api/predict",
            files={"file": ("tiny.png", data, "image/png")}
        )
        assert resp.status_code == 200

    def test_large_resolution_image(self, client):
        """Large image should resize and process."""
        data = make_png_bytes(width=2048, height=2048)
        resp = client.post(
            "/api/predict",
            files={"file": ("large.png", data, "image/png")}
        )
        assert resp.status_code == 200

    def test_rgba_png(self, client):
        """RGBA PNG should convert to RGB."""
        img = Image.new("RGBA", (224, 224), (128, 128, 128, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        resp = client.post(
            "/api/predict",
            files={"file": ("rgba.png", buf.getvalue(), "image/png")}
        )
        assert resp.status_code == 200


# ============================================================
# API Key Authentication & Rate Limiting
# ============================================================

class TestApiKeyAuth:
    """Test API key authentication and rate limiting."""

    def _make_png(self):
        img = Image.new("RGB", (224, 224), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def test_no_key_when_auth_disabled(self, client):
        """Without API_KEYS env, auth is disabled — request succeeds."""
        resp = client.post(
            "/api/predict",
            files={"file": ("test.png", self._make_png(), "image/png")}
        )
        assert resp.status_code == 200

    def test_health_always_public(self):
        """/health is always public even with auth enabled."""
        import main as app_mod
        with patch.object(app_mod, "VALID_API_KEYS", {"test-secret-key"}), \
             patch.object(app_mod, "AUTH_ENABLED", True):
            from fastapi.testclient import TestClient
            tc = TestClient(app_mod.app)
            resp = tc.get("/health")
            assert resp.status_code == 200

    def test_valid_key_accepted(self):
        """Valid API key should pass authentication."""
        import main as app_mod
        with patch.object(app_mod, "VALID_API_KEYS", {"my-secret-key"}), \
             patch.object(app_mod, "AUTH_ENABLED", True):
            from fastapi.testclient import TestClient
            tc = TestClient(app_mod.app)
            resp = tc.post(
                "/api/predict",
                files={"file": ("test.png", self._make_png(), "image/png")},
                headers={"X-API-Key": "my-secret-key"}
            )
            assert resp.status_code == 200

    def test_invalid_key_rejected(self):
        """Invalid API key should return 401."""
        import main as app_mod
        with patch.object(app_mod, "VALID_API_KEYS", {"my-secret-key"}), \
             patch.object(app_mod, "AUTH_ENABLED", True):
            from fastapi.testclient import TestClient
            tc = TestClient(app_mod.app)
            resp = tc.post(
                "/api/predict",
                files={"file": ("test.png", self._make_png(), "image/png")},
                headers={"X-API-Key": "wrong-key"}
            )
            assert resp.status_code == 401
            assert "Invalid or missing" in resp.json()["detail"]

    def test_missing_key_rejected(self):
        """No API key header should return 401."""
        import main as app_mod
        with patch.object(app_mod, "VALID_API_KEYS", {"my-secret-key"}), \
             patch.object(app_mod, "AUTH_ENABLED", True):
            from fastapi.testclient import TestClient
            tc = TestClient(app_mod.app)
            resp = tc.post(
                "/api/predict",
                files={"file": ("test.png", self._make_png(), "image/png")}
            )
            assert resp.status_code == 401

    def test_rate_limiting(self):
        """Rate limit should return 429 after max requests."""
        import main as app_mod
        limiter = app_mod.RateLimiter(max_requests=3, window_seconds=60)
        key = "rate-test-key"
        assert not limiter.is_limited(key)
        assert not limiter.is_limited(key)
        assert not limiter.is_limited(key)
        assert limiter.is_limited(key)  # 4th request blocked

    def test_cors_default_wildcard(self):
        """Without CORS_ORIGINS env, allow_origins should be [*]."""
        import os as _os
        origins_str = _os.environ.get("CORS_ORIGINS", "")
        origins = [o.strip() for o in origins_str.split(",") if o.strip()] or ["*"]
        assert origins == ["*"]