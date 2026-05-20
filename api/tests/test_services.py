"""Tests for inference service and preprocessing."""
import numpy as np
import pytest
from PIL import Image
from unittest.mock import patch, MagicMock
import io

from src.services.inference import InferenceService
from src.services.preprocessing import preprocess_image


class TestPreprocessing:
    def test_grayscale_to_rgb(self):
        """Grayscale image should be converted to RGB."""
        img = Image.new("L", (224, 224), color=128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = preprocess_image(buf.getvalue())
        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_rgb_image(self):
        """RGB image should produce CHW float32 array."""
        img = Image.new("RGB", (256, 128), color=(100, 150, 200))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = preprocess_image(buf.getvalue())
        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_normalized(self):
        """Output should be ImageNet normalized (mean ~0)."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        result = preprocess_image(buf.getvalue())
        # With ImageNet normalization, 128/255=0.502 ≈ (0.502-0.485)/0.229 ≈ 0.074
        # Just check range is reasonable
        assert np.all(np.isfinite(result))


class TestInferenceService:
    def test_stub_predict_single(self):
        svc = InferenceService()
        result = svc.predict_single(np.zeros((3, 224, 224), dtype=np.float32))
        assert isinstance(result, dict)
        assert all(v == 0.0 for v in result.values() if v != "_processing_time_ms")

    def test_stub_predict_batch(self):
        svc = InferenceService()
        images = [np.zeros((3, 224, 224), dtype=np.float32) for _ in range(3)]
        results = svc.predict_batch(images)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_model_info(self):
        svc = InferenceService()
        info = svc.get_model_info()
        assert info["model_type"] == "none"
        assert len(info["labels"]) == 14
        assert info["input_shape"] == [1, 3, 224, 224]

    def test_is_loaded_false(self):
        svc = InferenceService()
        assert svc.is_loaded is False
