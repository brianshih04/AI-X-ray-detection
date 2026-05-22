"""FastAPI Chest X-ray ONNX Inference Service with PostgreSQL."""
import logging
import io
import os
import time
import uuid
import struct
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import base64
import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
from PIL import Image
import cv2
import pydicom
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, JSON, text
from sqlalchemy.orm import declarative_base, sessionmaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax", "No_Finding",
]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:chestxpert123@postgres-svc.ai-xray.svc.cluster.local:5432/chestxpert"
)
Base = declarative_base()

class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=True)
    content_type = Column(String(50), nullable=True)
    file_size_kb = Column(Integer, nullable=True)
    top_prediction = Column(String(100), nullable=False)
    top_confidence = Column(Float, nullable=False)
    all_results = Column(JSON, nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

engine = None
SessionLocal = None
db_ready = False

def init_db():
    global engine, SessionLocal, db_ready
    try:
        engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=5, pool_pre_ping=True)
        SessionLocal = sessionmaker(bind=engine)
        Base.metadata.create_all(engine)
        db_ready = True
        logger.info("Database connected and tables created")
    except Exception as e:
        logger.warning(f"Database init failed: {e}")
        db_ready = False

session = None
cam_session = None
cam_weights = None

def load_model():
    global session, cam_session, cam_weights
    providers = ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model_path = "/app/models/best_model.onnx"
    session = ort.InferenceSession(model_path, sess_opts=sess_opts, providers=providers)
    logger.info(f"ONNX model loaded. Providers: {session.get_providers()}")

    # Load CAM model (same model with intermediate output added)
    cam_model_path = "/app/models/best_model_cam.onnx"
    cam_sess_opts = ort.SessionOptions()
    cam_sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    cam_session = ort.InferenceSession(cam_model_path, sess_opts=cam_sess_opts, providers=providers)
    # Extract classification weights for CAM computation
    onnx_model = onnx.load(cam_model_path)
    gemm_node = [n for n in onnx_model.graph.node if n.op_type == "Gemm"][0]
    for init in onnx_model.graph.initializer:
        if init.name == gemm_node.input[1]:
            cam_weights = numpy_helper.to_array(init)  # [15, 1024]
            break
    logger.info(f"CAM model loaded. Weights shape: {cam_weights.shape}")

ALLOWED_TYPES = {"image/png", "image/jpeg", "image/jpg", "application/dicom"}

# ---------------------------------------------------------------------------
# API Key Authentication & Rate Limiting
# ---------------------------------------------------------------------------
_api_keys_str = os.environ.get("API_KEYS", "")
VALID_API_KEYS: set[str] = {k.strip() for k in _api_keys_str.split(",") if k.strip()}
AUTH_ENABLED = bool(VALID_API_KEYS)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class RateLimiter:
    """Simple in-memory sliding-window rate limiter (stdlib only)."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._timestamps: dict[str, list[float]] = {}

    def is_limited(self, key: str) -> bool:
        now = time.time()
        timestamps = self._timestamps.get(key, [])
        cutoff = now - self.window
        # Prune old entries
        timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= self.max_requests:
            self._timestamps[key] = timestamps
            return True
        timestamps.append(now)
        self._timestamps[key] = timestamps
        return False


_rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


async def api_key_auth(api_key: str = Depends(_api_key_header)):
    """FastAPI dependency: validate API key and enforce rate limit."""
    if not AUTH_ENABLED:
        return None
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    if _rate_limiter.is_limited(api_key):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 100 requests per minute.",
        )
    return api_key

# ---------------------------------------------------------------------------

def is_dicom(data: bytes) -> bool:
    """Check if bytes look like a DICOM file."""
    # DICM magic at offset 128
    if len(data) > 132 and data[128:132] == b'DICM':
        return True
    # Some DICOM files don't have the preamble but start with group 0002
    if len(data) > 4:
        group = struct.unpack('<H', data[0:2])[0]
        if group == 0x0002:
            return True
    return False

def dicom_to_image(data: bytes) -> Image.Image:
    """Convert DICOM bytes to PIL Image."""
    ds = pydicom.dcmread(io.BytesIO(data))
    arr = ds.pixel_array
    if arr.ndim == 2:
        arr = arr.astype(np.float64)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr, mode='L').convert('RGB')
    elif arr.ndim == 3:
        if arr.shape[2] == 3:
            img = Image.fromarray(arr.astype(np.uint8), mode='RGB')
        elif arr.shape[2] == 4:
            img = Image.fromarray(arr[:, :, :3].astype(np.uint8), mode='RGB')
        else:
            img = Image.fromarray(arr[:, :, 0].astype(np.uint8), mode='L').convert('RGB')
    else:
        raise ValueError(f"Unsupported DICOM pixel array shape: {arr.shape}")
    return img

def preprocess(file_bytes):
    if is_dicom(file_bytes):
        img = dicom_to_image(file_bytes)
    else:
        img = Image.open(io.BytesIO(file_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return np.transpose(arr, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

@asynccontextmanager
async def lifespan(app):
    load_model()
    init_db()
    yield

app = FastAPI(title="Chest X-ray ONNX API", version="2.0.0", lifespan=lifespan)
_cors_origins_str = os.environ.get("CORS_ORIGINS", "")
_cors_origins = [o.strip() for o in _cors_origins_str.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": session is not None, "db_ready": db_ready,
            "providers": session.get_providers() if session else []}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5, _auth=Depends(api_key_auth)):
    """Predict chest X-ray findings with adjustable threshold.

    Args:
        threshold: Confidence threshold (0.0-1.0) for positive findings. Default 0.5.
                   All labels above this value are reported as positive findings.
    """
    threshold = max(0.0, min(1.0, threshold))
    if session is None:
        raise HTTPException(503, "Model not loaded")
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Use PNG, JPEG, or DICOM (.dcm)")
    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(413, "Image too large")
    try:
        inp = preprocess(file_bytes)
    except Exception as e:
        raise HTTPException(422, f"Failed to process image: {e}")
    start = time.perf_counter()
    logits = session.run(None, {"image": inp})[0][0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    elapsed_ms = (time.perf_counter() - start) * 1000
    results = [{"label": lbl, "confidence": round(float(p), 4)} for lbl, p in zip(LABELS, probs)]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    top = results[0]

    # Multi-label: find all positive findings above threshold
    positive_findings = [r for r in results if r["confidence"] >= threshold and r["label"] != "No_Finding"]

    # If nothing exceeds threshold, report top finding anyway
    if not positive_findings:
        positive_findings = [top]

    if db_ready:
        try:
            s = SessionLocal()
            s.add(PredictionRecord(
                filename=file.filename, content_type=file.content_type,
                file_size_kb=round(len(file_bytes)/1024, 1),
                top_prediction=top["label"], top_confidence=top["confidence"],
                all_results=results, processing_time_ms=round(elapsed_ms, 2)))
            s.commit()
            logger.info(f"Saved: {top['label']} ({top['confidence']:.2%}) | {len(positive_findings)} findings @ threshold={threshold}")
        except Exception as e:
            logger.warning(f"DB write failed: {e}")
        finally:
            s.close()
    return {"results": results, "top_prediction": top["label"],
            "top_confidence": top["confidence"], "threshold": threshold,
            "positive_findings": positive_findings,
            "processing_time_ms": round(elapsed_ms, 2)}

@app.get("/api/model/info")
async def model_info(_auth=Depends(api_key_auth)):
    return {"model": "DenseNet-121 (ONNX)", "labels": LABELS, "input_size": [1,3,224,224],
            "providers": session.get_providers() if session else []}

@app.get("/api/predictions")
async def list_predictions(limit: int = 20, _auth=Depends(api_key_auth)):
    if not db_ready:
        raise HTTPException(503, "Database not available")
    s = None
    try:
        s = SessionLocal()
        rows = s.execute(text(
            "SELECT id, filename, top_prediction, top_confidence, "
            "processing_time_ms, created_at FROM predictions "
            "ORDER BY created_at DESC LIMIT :lim"
        ), {"lim": limit}).fetchall()
        return [{
            "id": r[0], "filename": r[1], "top_prediction": r[2],
            "top_confidence": r[3], "processing_time_ms": r[4],
            "created_at": r[5].isoformat() if r[5] else None
        } for r in rows]
    except Exception as e:
        raise HTTPException(500, f"Query failed: {e}")
    finally:
        if s:
            s.close()

def generate_cam(inp, label_idx):
    """Generate CAM heatmap for a specific label."""
    results = cam_session.run(None, {"image": inp})
    feature_maps = results[1]  # [1, 1024, 7, 7]
    cam = np.zeros(feature_maps.shape[2:], dtype=np.float32)
    for i in range(feature_maps.shape[1]):
        cam += cam_weights[label_idx, i] * feature_maps[0, i, :, :]
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    return cam

@app.post("/api/gradcam")
async def gradcam(file: UploadFile = File(...), label: str = None, _auth=Depends(api_key_auth)):
    """Generate Grad-CAM heatmap overlay for an X-ray image."""
    if cam_session is None or cam_weights is None:
        raise HTTPException(503, "CAM model not loaded")
    if file.content_type and file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, "Use PNG, JPEG, or DICOM (.dcm)")
    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:
        raise HTTPException(413, "Image too large")
    try:
        inp = preprocess(file_bytes)
    except Exception as e:
        raise HTTPException(422, f"Failed to process image: {e}")

    start = time.perf_counter()
    logits = session.run(None, {"image": inp})[0][0]
    probs = 1.0 / (1.0 + np.exp(-logits))
    results = [{"label": lbl, "confidence": round(float(p), 4)} for lbl, p in zip(LABELS, probs)]
    results.sort(key=lambda x: x["confidence"], reverse=True)

    # Use specified label or top prediction
    target_label = label if label and label in LABELS else results[0]["label"]
    label_idx = LABELS.index(target_label)

    cam = generate_cam(inp, label_idx)
    cam_resized = np.array(Image.fromarray(cam).resize((224, 224), Image.BILINEAR))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    # Get original image (handle both DICOM and regular images)
    if is_dicom(file_bytes):
        orig = np.array(dicom_to_image(file_bytes).resize((224, 224)))
    else:
        orig = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB").resize((224, 224)))
    overlay = np.float32(heatmap) * 0.4 + np.float32(orig) * 0.6
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Encode to base64
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    heatmap_b64 = base64.b64encode(buf.getvalue()).decode()

    elapsed_ms = (time.perf_counter() - start) * 1000
    return {"results": results, "target_label": target_label,
            "heatmap": heatmap_b64, "processing_time_ms": round(elapsed_ms, 2)}
