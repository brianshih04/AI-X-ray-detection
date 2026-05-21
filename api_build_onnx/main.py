"""FastAPI Chest X-ray ONNX Inference Service with PostgreSQL."""
import logging
import io
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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

def load_model():
    global session
    providers = ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    model_path = "/app/models/best_model.onnx"
    session = ort.InferenceSession(model_path, sess_opts=sess_opts, providers=providers)
    logger.info(f"ONNX model loaded. Providers: {session.get_providers()}")

def preprocess(file_bytes):
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": session is not None, "db_ready": db_ready,
            "providers": session.get_providers() if session else []}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if session is None:
        raise HTTPException(503, "Model not loaded")
    if file.content_type and file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(400, "Use PNG or JPEG")
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
    results = [{"label": l, "confidence": round(float(p), 4)} for l, p in zip(LABELS, probs)]
    results.sort(key=lambda x: x["confidence"], reverse=True)
    top = results[0]
    if db_ready:
        try:
            s = SessionLocal()
            s.add(PredictionRecord(
                filename=file.filename, content_type=file.content_type,
                file_size_kb=round(len(file_bytes)/1024, 1),
                top_prediction=top["label"], top_confidence=top["confidence"],
                all_results=results, processing_time_ms=round(elapsed_ms, 2)))
            s.commit()
            logger.info(f"Saved: {top['label']} ({top['confidence']:.2%})")
        except Exception as e:
            logger.warning(f"DB write failed: {e}")
        finally:
            s.close()
    return {"results": results, "top_prediction": top["label"],
            "top_confidence": top["confidence"], "processing_time_ms": round(elapsed_ms, 2)}

@app.get("/api/model/info")
async def model_info():
    return {"model": "DenseNet-121 (ONNX)", "labels": LABELS, "input_size": [1,3,224,224],
            "providers": session.get_providers() if session else []}

@app.get("/api/predictions")
async def list_predictions(limit: int = 20):
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
