"""Model inference service — ONNX / TorchScript loader with batching support."""
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class InferenceService:
    """Singleton that loads ONNX or TorchScript model and runs inference."""

    def __init__(self):
        self._model = None
        self._model_type: Optional[str] = None  # "onnx" or "torchscript"
        self._labels = settings.LABELS
        self._input_shape: list[int] = [1, 3, 224, 224]
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_type(self) -> Optional[str]:
        return self._model_type

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def input_shape(self) -> list[int]:
        return self._input_shape

    def load_model(self) -> None:
        """Load model from disk. Tries ONNX first, then TorchScript."""
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            logger.warning("Model file not found at %s — inference will use dummy stub", model_path)
            self._loaded = False
            return

        suffix = model_path.suffix.lower()

        if suffix in (".onnx",):
            self._load_onnx(model_path)
        elif suffix in (".pt", ".pth", ".torchscript"):
            self._load_torchscript(model_path)
        else:
            # Try ONNX Runtime first
            try:
                self._load_onnx(model_path)
            except Exception:
                try:
                    self._load_torchscript(model_path)
                except Exception as e:
                    raise RuntimeError(f"Cannot load model from {model_path}: {e}")

    def _load_onnx(self, path: Path) -> None:
        import onnxruntime as ort
        providers = ["CPUExecutionProvider"]
        if "cuda" in settings.MODEL_DEVICE:
            providers.insert(0, "CUDAExecutionProvider")
        session = ort.InferenceSession(str(path), providers=providers)
        self._model = session
        self._model_type = "onnx"
        # Infer input shape from model
        inp = session.get_inputs()[0]
        self._input_shape = list(inp.shape)
        # Replace dynamic batch dim with 1 for reporting
        if self._input_shape[0] is None or self._input_shape[0] <= 0:
            self._input_shape[0] = 1
        self._loaded = True
        logger.info("Loaded ONNX model from %s  input_shape=%s  device=%s", path, self._input_shape, settings.MODEL_DEVICE)

    def _load_torchscript(self, path: Path) -> None:
        import torch
        device = torch.device(settings.MODEL_DEVICE)
        model = torch.jit.load(str(path), map_location=device)
        model.eval()
        self._model = model
        self._model_type = "torchscript"
        # Try to infer input shape
        try:
            dummy = torch.randn(1, 3, 224, 224, device=device)
            with torch.no_grad():
                _ = model(dummy)
            self._input_shape = [1, 3, 224, 224]
        except Exception:
            self._input_shape = [1, 3, 224, 224]
        self._loaded = True
        logger.info("Loaded TorchScript model from %s  device=%s", path, device)

    def predict_single(self, image_array: np.ndarray) -> dict[str, float]:
        """Run inference on a single preprocessed image array (NCHW, float32).

        Returns dict of {label: confidence_score}.
        """
        if not self._loaded:
            return self._dummy_predict()

        start = time.perf_counter()
        scores = self._run_inference(image_array[np.newaxis, ...])  # add batch dim
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = {label: float(score) for label, score in zip(self._labels, scores[0])}
        result["_processing_time_ms"] = elapsed_ms
        return result

    def predict_batch(self, images: list[np.ndarray]) -> list[dict[str, float]]:
        """Run inference on a batch of preprocessed image arrays."""
        if not images:
            return []
        if not self._loaded:
            return [self._dummy_predict() for _ in images]

        batch = np.stack(images, axis=0)
        start = time.perf_counter()
        scores = self._run_inference(batch)
        elapsed_ms = (time.perf_counter() - start) * 1000

        per_image_ms = elapsed_ms / len(images)
        results = []
        for i in range(len(images)):
            row = {label: float(score) for label, score in zip(self._labels, scores[i])}
            row["_processing_time_ms"] = per_image_ms
            results.append(row)
        return results

    def _run_inference(self, batch: np.ndarray) -> np.ndarray:
        """Core inference dispatch."""
        if self._model_type == "onnx":
            input_name = self._model.get_inputs()[0].name
            outputs = self._model.run(None, {input_name: batch.astype(np.float32)})
            raw = outputs[0]
            # Apply sigmoid for multi-label
            return 1.0 / (1.0 + np.exp(-raw))
        elif self._model_type == "torchscript":
            import torch
            tensor = torch.from_numpy(batch.astype(np.float32))
            with torch.no_grad():
                raw = self._model(tensor).cpu().numpy()
            return 1.0 / (1.0 + np.exp(-raw))
        else:
            raise RuntimeError("No model loaded")

    def _dummy_predict(self) -> dict[str, float]:
        """Return zero scores when no model is loaded (stub mode)."""
        return {label: 0.0 for label in self._labels}

    def get_model_info(self) -> dict:
        return {
            "model_version": settings.APP_VERSION,
            "model_type": self._model_type or "none",
            "labels": self._labels,
            "input_shape": self._input_shape,
            "device": settings.MODEL_DEVICE,
            "batch_size": settings.MODEL_BATCH_SIZE,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        }


# Singleton
inference_service = InferenceService()
