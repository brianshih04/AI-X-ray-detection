"""Core API endpoints: predict, patients, images, predictions, model info."""
import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from src.config import settings
from src.crud import (
    list_patients,
    get_patient_by_id,
    list_images_by_patient,
    get_image_by_id,
    get_image_labels,
    get_prediction_by_id,
    create_prediction,
)
from src.database import get_db
from src.schemas import (
    PredictResponse,
    PredictionResult,
    PatientListResponse,
    PatientOut,
    ImageListResponse,
    ImageOut,
    ImageDetailOut,
    ImageLabelOut,
    PredictionDetailOut,
    ModelInfoResponse,
    MessageResponse,
)
from src.services.auth import get_current_user_id
from src.services.inference import inference_service
from src.services.preprocessing import preprocess_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Core API"])
security = HTTPBearer(auto_error=False)


def _optional_user_id(creds: Optional[HTTPAuthorizationCredentials], db: Session) -> Optional[UUID]:
    """Extract user_id if token is provided, otherwise return None (anonymous)."""
    if creds is None:
        return None
    return get_current_user_id(creds.credentials, db)


# ─── POST /api/predict ──────────────────────────────

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Upload chest X-ray image and get classification results",
)
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image file (PNG/JPEG/DICOM)"),
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
):
    # Validate file type
    if file.content_type and file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/dicom"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported image format. Use PNG, JPEG, or DICOM.")

    # Read and preprocess
    file_bytes = await file.read()
    if len(file_bytes) > 50 * 1024 * 1024:  # 50 MB limit
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Image too large (max 50 MB).")

    try:
        image_array = preprocess_image(file_bytes)
    except Exception as e:
        logger.error("Image preprocessing failed: %s", e)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Failed to process image: {e}")

    # Run inference
    result = inference_service.predict_single(image_array)
    processing_time_ms = result.pop("_processing_time_ms", 0.0)

    # Filter to above-threshold results
    threshold = settings.CONFIDENCE_THRESHOLD
    results_list = [
        PredictionResult(label=label, confidence=round(score, 6))
        for label, score in result.items()
        if score >= threshold
    ]
    # Sort descending by confidence
    results_list.sort(key=lambda r: r.confidence, reverse=True)

    if not results_list:
        # At minimum return top-3
        sorted_labels = sorted(result.items(), key=lambda x: x[1], reverse=True)[:3]
        results_list = [PredictionResult(label=l, confidence=round(s, 6)) for l, s in sorted_labels]

    # Save to DB
    user_id = _optional_user_id(creds, db)
    prediction = create_prediction(
        db=db,
        user_id=user_id,
        image_id=None,  # No linked image record for direct uploads
        model_version=settings.APP_VERSION,
        results={r.label: r.confidence for r in results_list},
        processing_time_ms=processing_time_ms,
    )

    return PredictResponse(
        id=prediction.id,
        image_id=prediction.image_id,
        model_version=prediction.model_version,
        results=results_list,
        top_prediction=prediction.top_prediction,
        top_confidence=round(prediction.top_confidence, 6),
        processing_time_ms=round(processing_time_ms, 2),
        created_at=prediction.created_at,
    )


# ─── GET /api/patients ──────────────────────────────

@router.get("/patients", response_model=PatientListResponse, summary="List patients")
def get_patients(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
):
    patients, total = list_patients(db, page=page, page_size=page_size)
    return PatientListResponse(
        patients=patients,
        total=total,
        page=page,
        page_size=page_size,
    )


# ─── GET /api/patients/{id}/images ──────────────────

@router.get(
    "/patients/{patient_id}/images",
    response_model=ImageListResponse,
    summary="Get patient image records",
)
def get_patient_images(
    patient_id: UUID,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    patient = get_patient_by_id(db, patient_id)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    images, total = list_images_by_patient(db, patient_id, page=page, page_size=page_size)
    return ImageListResponse(images=images, total=total, page=page, page_size=page_size)


# ─── GET /api/patients/{id} ─────────────────────────

@router.get("/patients/{patient_id}", response_model=PatientOut, summary="Get patient by ID")
def get_patient(patient_id: UUID, db: Session = Depends(get_db)):
    patient = get_patient_by_id(db, patient_id)
    if not patient:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")
    return patient


# ─── GET /api/predictions/{id} ──────────────────────

@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionDetailOut,
    summary="Get prediction details by ID",
)
def get_prediction(prediction_id: UUID, db: Session = Depends(get_db)):
    prediction = get_prediction_by_id(db, prediction_id)
    if not prediction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")

    # Parse confidence_scores JSON
    import json
    try:
        scores = json.loads(prediction.confidence_scores) if prediction.confidence_scores else {}
    except json.JSONDecodeError:
        scores = {}

    results_list = [
        PredictionResult(label=l, confidence=round(s, 6))
        for l, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return PredictionDetailOut(
        id=prediction.id,
        user_id=prediction.user_id,
        image_id=prediction.image_id,
        model_version=prediction.model_version,
        results=results_list,
        top_prediction=prediction.top_prediction,
        top_confidence=round(prediction.top_confidence, 6) if prediction.top_confidence else 0.0,
        processing_time_ms=round(prediction.processing_time_ms, 2) if prediction.processing_time_ms else 0.0,
        created_at=prediction.created_at,
    )


# ─── GET /api/model/info ────────────────────────────

@router.get("/model/info", response_model=ModelInfoResponse, summary="Get model version and metrics")
def model_info():
    info = inference_service.get_model_info()
    return ModelInfoResponse(**info)
