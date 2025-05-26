from pydantic import BaseModel, Field, validator, constr, conint, confloat
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np
from enum import Enum
import re


class ModelType(str, Enum):
    """Supported face recognition models."""

    INSIGHTFACE = "insightface"
    VGGFACE2 = "vggface2"
    ENSEMBLE = "ensemble"


class FaceQuality(BaseModel):
    """Face quality assessment."""

    score: float = Field(..., ge=0, le=1, description="Overall quality score")
    brightness: float = Field(..., ge=0, le=1, description="Image brightness score")
    sharpness: float = Field(..., ge=0, le=1, description="Image sharpness score")
    pose: float = Field(..., ge=0, le=1, description="Face pose score")
    occlusion: float = Field(..., ge=0, le=1, description="Face occlusion score")


class FaceAttributes(BaseModel):
    """Face attributes."""

    age: int = Field(..., ge=0, le=100, description="Estimated age")
    gender: str = Field(..., regex="^[MF]$", description="Gender (M/F)")
    emotion: str = Field(
        ...,
        regex="^(angry|disgust|fear|happy|sad|surprise|neutral)$",
        description="Detected emotion",
    )


class FaceMatch(BaseModel):
    """Face recognition match."""

    name: str = Field(..., min_length=1, description="Person name")
    confidence: float = Field(..., ge=0, le=1, description="Match confidence score")
    model: ModelType = Field(..., description="Model used for recognition")


class FaceDetection(BaseModel):
    """Face detection result."""

    bbox: List[float] = Field(
        ..., min_items=4, max_items=4, description="Bounding box [x1, y1, x2, y2]"
    )
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    quality: FaceQuality = Field(..., description="Face quality assessment")
    attributes: Optional[FaceAttributes] = Field(None, description="Face attributes")
    matches: Optional[List[FaceMatch]] = Field(None, description="Recognition matches")


class RegisterFaceRequest(BaseModel):
    """Request model for face registration."""

    name: constr(min_length=1, max_length=100, strip_whitespace=True) = Field(
        ..., description="Name of the person"
    )
    model: ModelType = Field(
        default=ModelType.ENSEMBLE, description="Model to use for face recognition"
    )
    overwrite: bool = Field(
        default=False, description="Whether to overwrite existing embeddings"
    )

    @validator("name")
    def validate_name(cls, v):
        """Validate person name."""
        if not re.match(r"^[a-zA-Z0-9_\- ]+$", v):
            raise ValueError(
                "Name can only contain letters, numbers, spaces, hyphens and underscores"
            )
        return v


class RecognizeFaceRequest(BaseModel):
    """Request model for face recognition."""

    model: ModelType = Field(
        default=ModelType.ENSEMBLE, description="Model to use for face recognition"
    )
    min_confidence: confloat(ge=0, le=1) = Field(
        default=0.6, description="Minimum confidence threshold"
    )
    include_attributes: bool = Field(
        default=False, description="Whether to include face attributes"
    )
    max_faces: conint(ge=1, le=20) = Field(
        default=5, description="Maximum number of faces to detect"
    )


class RegisterFaceResponse(BaseModel):
    """Response model for face registration."""

    success: bool = Field(..., description="Operation success status")
    data: Dict[str, Any] = Field(
        ..., description="Response data containing embedding info"
    )
    error: Optional[Dict[str, Any]] = Field(
        None, description="Error information if success is False"
    )


class RecognizeFaceResponse(BaseModel):
    """Response model for face recognition."""

    success: bool = Field(..., description="Operation success status")
    data: Dict[str, Any] = Field(
        ..., description="Response data containing recognition results"
    )
    error: Optional[Dict[str, Any]] = Field(
        None, description="Error information if success is False"
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, Dict[str, Any]] = Field(
        ..., description="Component health status"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Health check timestamp"
    )


class ModelInfo(BaseModel):
    """Model information."""

    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    type: str = Field(..., description="Model type")
    input_shape: List[int] = Field(..., description="Input tensor shape")
    framework: str = Field(..., description="Deep learning framework")
    provider: str = Field(..., description="Model provider")
    license: str = Field(..., description="Model license")
    last_updated: datetime = Field(..., description="Last update timestamp")
    metrics: Dict[str, float] = Field(..., description="Model performance metrics")


class EmbeddingVector(BaseModel):
    """Face embedding vector."""

    vector: List[float] = Field(..., description="Embedding vector")

    @validator("vector")
    def validate_vector(cls, v):
        """Validate embedding vector."""
        if len(v) not in [512, 2048]:  # Common embedding sizes
            raise ValueError("Invalid embedding vector size")

        # Convert to numpy for normalization
        vector = np.array(v)
        norm = np.linalg.norm(vector)

        if norm == 0 or np.isnan(norm):
            raise ValueError("Invalid embedding vector (zero or NaN values)")

        # Normalize vector
        vector = vector / norm

        return vector.tolist()


class ErrorResponse(BaseModel):
    """Error response model."""

    success: bool = Field(default=False, description="Operation success status")
    error: Dict[str, Any] = Field(..., description="Error details")

    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Invalid input data",
                    "details": {"field": "name", "error": "Field required"},
                },
            }
        }
