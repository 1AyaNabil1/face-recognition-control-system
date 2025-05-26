from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class MobileRecognizeResponse(BaseModel):
    """Lightweight response for mobile face recognition."""

    success: bool
    name: Optional[str] = None
    confidence: Optional[float] = None
    face_box: Optional[List[int]] = None
    quality: Optional[float] = None
    matched: bool = False
    inference_time_ms: Optional[int] = None
    liveness_status: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class MobileRegisterRequest(BaseModel):
    """Request model for mobile registration."""

    name: str = Field(..., min_length=1, max_length=100)
    metadata: Optional[Dict] = None


class MobileRegisterResponse(BaseModel):
    """Lightweight response for mobile registration."""

    success: bool
    person_id: Optional[str] = None
    quality: Optional[float] = None
    inference_time_ms: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class MobileVerifyResponse(BaseModel):
    """Lightweight response for mobile verification."""

    success: bool
    matched: bool = False
    confidence: Optional[float] = None
    quality: Optional[float] = None
    inference_time_ms: Optional[int] = None
    liveness_status: Optional[str] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class MobileHealthResponse(BaseModel):
    """Lightweight health check response for mobile."""

    status: str
    api_version: str
    models_ready: bool
    last_updated: datetime
