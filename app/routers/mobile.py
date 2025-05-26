from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import numpy as np
import cv2
import time
from typing import Optional
import structlog
from datetime import datetime
import uuid

from app.models.mobile import (
    MobileRecognizeResponse,
    MobileRegisterRequest,
    MobileRegisterResponse,
    MobileVerifyResponse,
    MobileHealthResponse,
)
from app.core.database import get_db
from app.repositories.face_repository import FaceRepository
from app.models.face_storage import PersonCreate, FaceEmbeddingCreate
from app.recognition.face_recognition_engine import ProductionFaceRecognitionEngine
from app.core.config import settings

# Initialize logger
logger = structlog.get_logger(__name__)

# Initialize security (placeholder for future auth)
security = HTTPBearer(auto_error=False)

# Create router
router = APIRouter(prefix="/api/v1/mobile", tags=["Mobile API"])

# Initialize face engine (imported from main app instance)
face_engine: ProductionFaceRecognitionEngine = None


def init_face_engine(engine: ProductionFaceRecognitionEngine):
    """Initialize face engine from main app instance."""
    global face_engine
    face_engine = engine


async def process_image_file(file: UploadFile) -> np.ndarray:
    """Process uploaded image file with mobile-specific handling."""
    try:
        # Validate content type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error_code": "INVALID_CONTENT_TYPE",
                    "error_message": "File must be an image (JPEG or PNG)",
                },
            )

        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "error_code": "INVALID_IMAGE",
                    "error_message": "Could not decode image file",
                },
            )

        # Handle EXIF orientation
        # TODO: Add EXIF handling if needed

        return image

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing image file", error=str(e))
        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error_code": "IMAGE_PROCESSING_ERROR",
                "error_message": "Error processing image file",
            },
        )


@router.post("/recognize", response_model=MobileRecognizeResponse)
async def mobile_recognize(
    file: UploadFile = File(...),
    token: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """
    Mobile-optimized face recognition endpoint.
    Accepts a single image file and returns lightweight recognition results.
    """
    try:
        start_time = time.time()

        # Process image
        image = await process_image_file(file)

        # Process face
        face_data = await face_engine.process_face(image)
        if not face_data:
            return MobileRecognizeResponse(
                success=False,
                error_code="NO_FACE_DETECTED",
                error_message="No face detected in image",
                inference_time_ms=int((time.time() - start_time) * 1000),
            )

        # Check liveness
        if not face_data.get("is_real"):
            return MobileRecognizeResponse(
                success=False,
                error_code="SPOOF_DETECTED",
                error_message="Potential spoof detected",
                quality=face_data["quality"]["score"],
                liveness_status="spoof",
                inference_time_ms=int((time.time() - start_time) * 1000),
            )

        # Find matches
        async with get_db() as db:
            repo = FaceRepository(db)
            similar = await repo.find_similar_embeddings(
                face_data["embedding"], threshold=0.6, limit=1
            )

        # Prepare response
        response = MobileRecognizeResponse(
            success=True,
            matched=bool(similar),
            quality=face_data["quality"]["score"],
            face_box=[int(x) for x in face_data["bbox"]],
            liveness_status="real",
            inference_time_ms=int((time.time() - start_time) * 1000),
        )

        if similar:
            match, score = similar[0]
            response.name = match.person.name
            response.confidence = score

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Recognition error", error=str(e))
        return MobileRecognizeResponse(
            success=False,
            error_code="INTERNAL_ERROR",
            error_message="Internal server error",
            inference_time_ms=int((time.time() - start_time) * 1000),
        )


@router.post("/register", response_model=MobileRegisterResponse)
async def mobile_register(
    file: UploadFile = File(...),
    name: str = Form(...),
    metadata: Optional[str] = Form(None),
    token: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """
    Mobile-optimized face registration endpoint.
    Accepts a single image file and person details.
    """
    try:
        start_time = time.time()

        # Process image
        image = await process_image_file(file)

        # Process face
        face_data = await face_engine.process_face(image)
        if not face_data:
            return MobileRegisterResponse(
                success=False,
                error_code="NO_FACE_DETECTED",
                error_message="No face detected in image",
                inference_time_ms=int((time.time() - start_time) * 1000),
            )

        # Check liveness
        if not face_data.get("is_real"):
            return MobileRegisterResponse(
                success=False,
                error_code="SPOOF_DETECTED",
                error_message="Potential spoof detected",
                quality=face_data["quality"]["score"],
                inference_time_ms=int((time.time() - start_time) * 1000),
            )

        # Create person and embedding
        async with get_db() as db:
            repo = FaceRepository(db)

            # Check for duplicates
            similar = await repo.find_similar_embeddings(
                face_data["embedding"], threshold=0.9
            )
            if similar:
                return MobileRegisterResponse(
                    success=False,
                    error_code="DUPLICATE_FACE",
                    error_message="Face already registered",
                    inference_time_ms=int((time.time() - start_time) * 1000),
                )

            # Create person
            person = await repo.create_person(
                PersonCreate(name=name, metadata={"source": "mobile"})
            )

            # Create embedding
            embedding = await repo.create_face_embedding(
                FaceEmbeddingCreate(
                    person_id=person.id,
                    embedding=face_data["embedding"],
                    model_name="facenet",
                    quality_score=face_data["quality"]["score"],
                    sharpness=face_data["quality"]["sharpness"],
                    brightness=face_data["quality"]["brightness"],
                    contrast=face_data["quality"]["contrast"],
                    face_size=face_data["quality"]["face_size"],
                    is_frontal=face_data["quality"]["is_frontal"],
                    metadata={
                        "source": "mobile",
                        "attributes": face_data.get("attributes", {}),
                    },
                )
            )

        return MobileRegisterResponse(
            success=True,
            person_id=str(person.id),
            quality=face_data["quality"]["score"],
            inference_time_ms=int((time.time() - start_time) * 1000),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Registration error", error=str(e))
        return MobileRegisterResponse(
            success=False,
            error_code="INTERNAL_ERROR",
            error_message="Internal server error",
            inference_time_ms=int((time.time() - start_time) * 1000),
        )


@router.post("/verify", response_model=MobileVerifyResponse)
async def mobile_verify(
    file: UploadFile = File(...),
    person_id: str = Form(...),
    token: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """
    Mobile-optimized face verification endpoint.
    Accepts a single image file and person ID for 1:1 verification.
    """
    try:
        start_time = time.time()

        # Process image
        image = await process_image_file(file)

        # Process face
        face_data = await face_engine.process_face(image)
        if not face_data:
            return MobileVerifyResponse(
                success=False,
                error_code="NO_FACE_DETECTED",
                error_message="No face detected in image",
                inference_time_ms=int((time.time() - start_time) * 1000),
            )

        # Check liveness
        if not face_data.get("is_real"):
            return MobileVerifyResponse(
                success=False,
                error_code="SPOOF_DETECTED",
                error_message="Potential spoof detected",
                quality=face_data["quality"]["score"],
                liveness_status="spoof",
                inference_time_ms=int((time.time() - start_time) * 1000),
            )

        # Get person's embeddings
        try:
            person_uuid = uuid.UUID(person_id)
        except ValueError:
            return MobileVerifyResponse(
                success=False,
                error_code="INVALID_PERSON_ID",
                error_message="Invalid person ID format",
                inference_time_ms=int((time.time() - start_time) * 1000),
            )

        async with get_db() as db:
            repo = FaceRepository(db)
            stored_embeddings = await repo.get_person_embeddings(person_uuid)

            if not stored_embeddings:
                return MobileVerifyResponse(
                    success=False,
                    error_code="PERSON_NOT_FOUND",
                    error_message="Person not found or no embeddings stored",
                    inference_time_ms=int((time.time() - start_time) * 1000),
                )

            # Find best match
            query_embedding = np.array(face_data["embedding"])
            best_score = 0.0

            for emb in stored_embeddings:
                stored_embedding = np.array(emb.embedding)
                score = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                best_score = max(best_score, score)

            return MobileVerifyResponse(
                success=True,
                matched=best_score >= 0.6,
                confidence=float(best_score),
                quality=face_data["quality"]["score"],
                liveness_status="real",
                inference_time_ms=int((time.time() - start_time) * 1000),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Verification error", error=str(e))
        return MobileVerifyResponse(
            success=False,
            error_code="INTERNAL_ERROR",
            error_message="Internal server error",
            inference_time_ms=int((time.time() - start_time) * 1000),
        )


@router.get("/health", response_model=MobileHealthResponse)
async def mobile_health():
    """
    Mobile-optimized health check endpoint.
    Returns minimal status information needed by mobile clients.
    """
    try:
        models_ready = face_engine is not None and face_engine.get_status()["is_ready"]
        return MobileHealthResponse(
            status="healthy" if models_ready else "degraded",
            api_version=settings.api.VERSION,
            models_ready=models_ready,
            last_updated=datetime.utcnow(),
        )
    except Exception as e:
        logger.error("Health check error", error=str(e))
        return MobileHealthResponse(
            status="unhealthy",
            api_version=settings.api.VERSION,
            models_ready=False,
            last_updated=datetime.utcnow(),
        )
