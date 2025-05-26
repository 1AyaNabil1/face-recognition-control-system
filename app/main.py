from fastapi import FastAPI, Request, Response, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from prometheus_client import make_asgi_app
import uvicorn
import structlog
from typing import Dict, Any, List
import os
import asyncio
from datetime import datetime
import numpy as np
import cv2
from pydantic import BaseModel, Field
import base64
import time

from app.core.config import settings, validate_settings
from app.core.middleware import (
    RequestContextMiddleware,
    MetricsMiddleware,
    SecurityMiddleware,
    ErrorHandlingMiddleware,
    LoggingMiddleware,
)
from app.core.error_handlers import setup_error_handlers
from app.core.logging_config import setup_logging
from app.core.routes import api_router
from app.core.database import init_db, run_migrations, engine, get_db
from app.core.cache import RedisCache
from app.recognition.model_manager import ModelManager
from app.recognition.face_recognition_engine import ProductionFaceRecognitionEngine
from app.models.face_storage import (
    RegisterRequest,
    RegisterResponse,
    VerifyRequest,
    VerifyResponse,
    VerifyMatch,
    PersonCreate,
    FaceEmbeddingCreate,
)
from app.repositories.face_repository import FaceRepository
from app.routers.mobile import router as mobile_router, init_face_engine

# Initialize logger
logger = structlog.get_logger(__name__)

# Global instances
redis_cache: RedisCache = None
model_manager: ModelManager = None
face_engine: ProductionFaceRecognitionEngine = None


# API Models
class RegisterRequest(BaseModel):
    name: str = Field(..., description="Name of the person to register")
    images: List[str] = Field(..., description="Base64 encoded face images")


class VerifyRequest(BaseModel):
    name: str = Field(..., description="Name of the person to verify against")
    image: str = Field(..., description="Base64 encoded face image to verify")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    # Validate settings
    validate_settings()

    # Configure logging
    setup_logging(
        log_level=settings.monitoring.LOG_LEVEL, log_file=settings.monitoring.LOG_FILE
    )

    # Initialize Sentry
    if settings.monitoring.SENTRY_DSN:
        sentry_sdk.init(
            dsn=settings.monitoring.SENTRY_DSN,
            environment=settings.monitoring.SENTRY_ENVIRONMENT,
            traces_sample_rate=settings.monitoring.SENTRY_TRACES_SAMPLE_RATE,
        )

    # Create FastAPI app
    app = FastAPI(
        title=settings.api.TITLE,
        description=settings.api.DESCRIPTION,
        version=settings.api.VERSION,
        docs_url=settings.api.DOCS_URL,
        redoc_url=settings.api.REDOC_URL,
        openapi_url=settings.api.OPENAPI_URL,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add trusted host middleware
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=settings.security.TRUSTED_HOSTS
    )

    # Add custom middleware
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)

    # Add Sentry middleware if configured
    if settings.monitoring.SENTRY_DSN:
        app.add_middleware(SentryAsgiMiddleware)

    # Set up error handlers
    setup_error_handlers(app)

    # Add prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Add API routes
    app.include_router(api_router, prefix=f"/api/{settings.api.VERSION}")
    app.include_router(mobile_router)  # Mobile endpoints are under /api/v1/mobile/

    # Face Recognition API Endpoints
    @app.post("/recognize", tags=["Face Recognition"])
    async def recognize_faces(file: UploadFile = File(...)):
        """
        Recognize faces in an image with full metadata.

        Returns:
        - List of detected faces with recognition results
        - Each face includes: bbox, quality, attributes, matches
        """
        try:
            # Read and decode image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image format")

            # Get cached embeddings
            cache_key = "known_faces"
            cached_data = await redis_cache.get(cache_key)

            if not cached_data:
                async with get_db() as db:
                    # Fetch from database and cache
                    known_faces = await db.fetch_all_embeddings()
                    await redis_cache.set(cache_key, known_faces, ttl=3600)
            else:
                known_faces = cached_data

            # Process image
            results = await face_engine.recognize_faces(
                image,
                known_embeddings=[face["embedding"] for face in known_faces],
                known_names=[face["name"] for face in known_faces],
                include_attributes=True,
            )

            return {"faces": results}

        except Exception as e:
            logger.error("Recognition error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/register", tags=["Face Recognition"], response_model=RegisterResponse)
    async def register_person(request: RegisterRequest):
        """
        Register a new person with multiple face images.

        Features:
        - Multiple image support
        - Quality and liveness checks
        - Duplicate detection
        - Async database storage
        - Optional image storage
        - Redis caching
        """
        try:
            start_time = time.time()
            failed_images = []

            # Initialize repository
            async with get_db() as db:
                repo = FaceRepository(db, redis_cache)

                # Create person record
                person = await repo.create_person(
                    PersonCreate(name=request.name, metadata=request.metadata)
                )

                # Process each image
                embeddings = []
                for idx, image_b64 in enumerate(request.images):
                    try:
                        # Decode image
                        nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
                        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        if image is None:
                            failed_images.append(
                                {"index": idx, "error": "Invalid image format"}
                            )
                            continue

                        # Process face
                        face_data = await face_engine.process_face(image)
                        if not face_data:
                            failed_images.append(
                                {"index": idx, "error": "No face detected"}
                            )
                            continue

                        if not face_data.get("is_real"):
                            failed_images.append(
                                {"index": idx, "error": "Spoof detected"}
                            )
                            continue

                        # Check for duplicates
                        similar = await repo.find_similar_embeddings(
                            face_data["embedding"],
                            threshold=0.9,  # High threshold for duplicates
                        )
                        if similar:
                            failed_images.append(
                                {
                                    "index": idx,
                                    "error": "Duplicate face detected",
                                    "similar_to": str(similar[0][0].id),
                                }
                            )
                            continue

                        # Create embedding record
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
                                    "attributes": face_data.get("attributes", {}),
                                    "processing_time": face_data["processing_time"],
                                },
                            )
                        )
                        embeddings.append(embedding)

                    except Exception as e:
                        logger.error("Error processing image", error=str(e))
                        failed_images.append({"index": idx, "error": str(e)})

                return RegisterResponse(
                    person=person,
                    embeddings=embeddings,
                    failed_images=failed_images,
                    processing_time=time.time() - start_time,
                )

        except Exception as e:
            logger.error("Registration error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/verify", tags=["Face Recognition"], response_model=VerifyResponse)
    async def verify_face(request: VerifyRequest):
        """
        Verify if a face image matches a registered person.

        Features:
        - 1:1 verification
        - Multiple similarity metrics
        - Quality assessment
        - Configurable threshold
        - Cached embedding lookup
        """
        try:
            start_time = time.time()

            # Decode image
            nparr = np.frombuffer(base64.b64decode(request.image), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image format")

            # Process face
            face_data = await face_engine.process_face(image)
            if not face_data:
                raise HTTPException(status_code=400, detail="No face detected")

            if not face_data.get("is_real"):
                raise HTTPException(status_code=400, detail="Spoof detected")

            # Get person's embeddings
            async with get_db() as db:
                repo = FaceRepository(db, redis_cache)
                stored_embeddings = await repo.get_person_embeddings(request.person_id)

                if not stored_embeddings:
                    raise HTTPException(
                        status_code=404,
                        detail="Person not found or no embeddings stored",
                    )

                # Find matches
                matches = []
                query_embedding = np.array(face_data["embedding"])

                for emb in stored_embeddings:
                    stored_embedding = np.array(emb.embedding)

                    if request.metric == "cosine":
                        score = np.dot(query_embedding, stored_embedding) / (
                            np.linalg.norm(query_embedding)
                            * np.linalg.norm(stored_embedding)
                        )
                        distance = 1 - score
                    else:  # L2
                        distance = np.linalg.norm(query_embedding - stored_embedding)
                        score = 1 / (1 + distance)

                    if score >= request.threshold:
                        matches.append(
                            VerifyMatch(
                                embedding_id=emb.id,
                                score=float(score),
                                distance=float(distance),
                                quality_score=emb.quality_score,
                                metadata=emb.metadata,
                            )
                        )

                # Sort matches by score
                matches.sort(key=lambda x: x.score, reverse=True)

                return VerifyResponse(
                    match=len(matches) > 0,
                    confidence_score=matches[0].score if matches else 0.0,
                    distance_metric=request.metric,
                    matches=matches,
                    face_quality=face_data["quality"],
                    processing_time=time.time() - start_time,
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Verification error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        components = {
            "api": {"status": "healthy", "timestamp": datetime.utcnow().isoformat()},
            "database": {
                "status": "unknown",
                "timestamp": datetime.utcnow().isoformat(),
            },
            "redis": {"status": "unknown", "timestamp": datetime.utcnow().isoformat()},
            "model": {"status": "unknown", "timestamp": datetime.utcnow().isoformat()},
        }

        # Check database
        try:
            async with get_db() as db:
                await db.execute("SELECT 1")
            components["database"]["status"] = "healthy"
        except Exception as e:
            components["database"]["status"] = "unhealthy"
            components["database"]["error"] = str(e)

        # Check Redis
        if redis_cache:
            try:
                if await redis_cache.ping():
                    components["redis"]["status"] = "healthy"
                else:
                    components["redis"]["status"] = "unhealthy"
                    components["redis"]["error"] = "Redis ping failed"
            except Exception as e:
                components["redis"]["status"] = "unhealthy"
                components["redis"]["error"] = str(e)

        # Check model service
        if face_engine:
            try:
                model_status = face_engine.get_status()
                if model_status["is_ready"]:
                    components["model"].update(
                        {
                            "status": "healthy",
                            "device": model_status["device"],
                            "models": model_status["models"],
                        }
                    )
                else:
                    components["model"]["status"] = "degraded"
                    components["model"]["error"] = "Some models not loaded"
            except Exception as e:
                components["model"]["status"] = "unhealthy"
                components["model"]["error"] = str(e)

        # Calculate overall status
        overall_status = "healthy"
        for component in components.values():
            if component["status"] != "healthy":
                overall_status = "degraded"
                break

        return {
            "status": overall_status,
            "version": settings.api.VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "components": components,
        }

    # Add startup event
    @app.on_event("startup")
    async def startup_event():
        """Run startup tasks."""
        logger.info(
            "Starting application",
            version=settings.api.VERSION,
            environment=settings.ENV,
        )

        # Initialize components
        try:
            # Initialize database
            await init_db()
            await run_migrations()
            logger.info("Database initialized successfully")

            # Initialize Redis
            global redis_cache
            redis_cache = RedisCache()
            if not await redis_cache.ping():
                raise Exception("Redis connection failed")
            logger.info("Redis cache initialized successfully")

            # Initialize face recognition engine
            global face_engine
            face_engine = ProductionFaceRecognitionEngine()
            init_face_engine(face_engine)  # Initialize mobile router's face engine
            logger.info("Face recognition engine initialized successfully")

            # Initialize model manager
            global model_manager
            model_manager = ModelManager()
            await model_manager.load_models()
            logger.info("Model manager initialized successfully")

        except Exception as e:
            logger.error("Startup failed", error=str(e))
            raise

    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Run shutdown tasks."""
        logger.info("Shutting down application")

        try:
            # Close database connections
            await engine.dispose()
            logger.info("Database connections closed")

            # Clear Redis cache
            if redis_cache:
                await redis_cache.clear()
                await redis_cache.close()
                logger.info("Redis cache cleared and closed")

            # Unload models
            if model_manager:
                await model_manager.unload_models()
                logger.info("Models unloaded successfully")

        except Exception as e:
            logger.error("Shutdown failed", error=str(e))

    return app


# Create application instance
app = create_app()

if __name__ == "__main__":
    # Run with uvicorn if called directly
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        workers=os.cpu_count() * 2 + 1,
        log_config=None,  # Use custom logging config
    )
