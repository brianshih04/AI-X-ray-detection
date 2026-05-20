"""Image preprocessing pipeline for chest X-ray inference."""
import io
from typing import Optional

import numpy as np
from PIL import Image

from src.config import settings


# ImageNet normalization constants
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Standard input size for DenseNet / EfficientNet
_TARGET_SIZE = (224, 224)


def preprocess_image(
    file_bytes: bytes,
    target_size: tuple[int, int] = _TARGET_SIZE,
) -> np.ndarray:
    """Preprocess raw image bytes to model-ready numpy array (NCHW, float32).

    Pipeline:
    1. Decode bytes → PIL Image (grayscale or RGB)
    2. Convert grayscale → RGB (3 channels)
    3. Resize to target_size (bilinear)
    4. Convert to float32 [0, 1]
    5. ImageNet normalize
    6. Rearrange to CHW
    """
    img = Image.open(io.BytesIO(file_bytes))

    # Convert grayscale or palette to RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize
    img = img.resize(target_size, Image.BILINEAR)

    # To numpy float32 [0, 1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # ImageNet normalize
    arr = (arr - _MEAN) / _STD

    # HWC → CHW
    arr = np.transpose(arr, (2, 0, 1))

    return arr
