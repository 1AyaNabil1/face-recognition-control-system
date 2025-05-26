from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    ForeignKey,
    JSON,
    DateTime,
    Boolean,
    LargeBinary,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from typing import List, Optional
from pydantic import BaseModel, Field, constr
import numpy as np

from app.core.database import Base


class Person(Base):
    """Person record in the database."""

    __tablename__ = "persons"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationships
    face_embeddings = relationship(
        "FaceEmbedding", back_populates="person", cascade="all, delete-orphan"
    )


class FaceEmbedding(Base):
    """Face embedding storage."""

    __tablename__ = "face_embeddings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    person_id = Column(UUID(as_uuid=True), ForeignKey("persons.id"), nullable=False)
    embedding = Column(ARRAY(Float), nullable=False)
    model_name = Column(String, nullable=False)
    quality_score = Column(Float, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Quality metrics
    sharpness = Column(Float)
    brightness = Column(Float)
    contrast = Column(Float)
    face_size = Column(Float)
    is_frontal = Column(Boolean)

    # Optional image storage
    image_hash = Column(String, nullable=True)
    image_data = Column(LargeBinary, nullable=True)

    # Metadata
    metadata = Column(JSON, nullable=True)

    # Relationships
    person = relationship("Person", back_populates="face_embeddings")


# Pydantic models for API
class PersonCreate(BaseModel):
    """Request model for creating a person."""

    name: constr(min_length=1, max_length=100)
    metadata: Optional[dict] = None


class PersonResponse(BaseModel):
    """Response model for person data."""

    id: uuid.UUID
    name: str
    created_at: datetime
    updated_at: datetime
    metadata: Optional[dict] = None
    is_active: bool

    class Config:
        orm_mode = True


class FaceEmbeddingCreate(BaseModel):
    """Request model for creating a face embedding."""

    person_id: uuid.UUID
    embedding: List[float]
    model_name: str
    quality_score: float
    sharpness: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    face_size: Optional[float] = None
    is_frontal: Optional[bool] = None
    image_hash: Optional[str] = None
    metadata: Optional[dict] = None


class FaceEmbeddingResponse(BaseModel):
    """Response model for face embedding data."""

    id: uuid.UUID
    person_id: uuid.UUID
    model_name: str
    quality_score: float
    created_at: datetime
    metadata: Optional[dict] = None
    is_active: bool

    class Config:
        orm_mode = True


class RegisterRequest(BaseModel):
    """Request model for face registration."""

    name: constr(min_length=1, max_length=100)
    images: List[str] = Field(..., description="List of base64 encoded images")
    metadata: Optional[dict] = None
    store_images: bool = False


class RegisterResponse(BaseModel):
    """Response model for face registration."""

    person: PersonResponse
    embeddings: List[FaceEmbeddingResponse]
    failed_images: List[dict]
    processing_time: float


class VerifyRequest(BaseModel):
    """Request model for face verification."""

    person_id: uuid.UUID
    image: str = Field(..., description="Base64 encoded image")
    threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0)
    metric: str = Field("cosine", pattern="^(cosine|l2)$")


class VerifyMatch(BaseModel):
    """Match result for verification."""

    embedding_id: uuid.UUID
    score: float
    distance: float
    quality_score: float
    metadata: Optional[dict] = None


class VerifyResponse(BaseModel):
    """Response model for face verification."""

    match: bool
    confidence_score: float
    distance_metric: str
    matches: List[VerifyMatch]
    face_quality: dict
    processing_time: float
