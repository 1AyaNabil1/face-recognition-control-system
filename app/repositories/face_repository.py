from typing import List, Optional, Dict, Tuple
import uuid
import numpy as np
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
import structlog
import time
from datetime import datetime

from app.models.face_storage import (
    Person,
    FaceEmbedding,
    PersonCreate,
    FaceEmbeddingCreate,
)
from app.core.cache import RedisCache

# Initialize logger
logger = structlog.get_logger(__name__)


class FaceRepository:
    """Repository for face storage operations."""

    def __init__(self, session: AsyncSession, cache: Optional[RedisCache] = None):
        self.session = session
        self.cache = cache

    async def create_person(self, person_data: PersonCreate) -> Person:
        """Create a new person record."""
        try:
            person = Person(name=person_data.name, metadata=person_data.metadata)
            self.session.add(person)
            await self.session.commit()
            await self.session.refresh(person)
            return person
        except Exception as e:
            await self.session.rollback()
            logger.error("Error creating person", error=str(e))
            raise

    async def get_person(self, person_id: uuid.UUID) -> Optional[Person]:
        """Get person by ID with their embeddings."""
        try:
            stmt = (
                select(Person)
                .options(selectinload(Person.face_embeddings))
                .where(and_(Person.id == person_id, Person.is_active == True))
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Error getting person", error=str(e))
            raise

    async def create_face_embedding(
        self, embedding_data: FaceEmbeddingCreate
    ) -> FaceEmbedding:
        """Create a new face embedding."""
        try:
            embedding = FaceEmbedding(
                person_id=embedding_data.person_id,
                embedding=embedding_data.embedding,
                model_name=embedding_data.model_name,
                quality_score=embedding_data.quality_score,
                sharpness=embedding_data.sharpness,
                brightness=embedding_data.brightness,
                contrast=embedding_data.contrast,
                face_size=embedding_data.face_size,
                is_frontal=embedding_data.is_frontal,
                image_hash=embedding_data.image_hash,
                metadata=embedding_data.metadata,
            )
            self.session.add(embedding)
            await self.session.commit()
            await self.session.refresh(embedding)

            # Invalidate cache
            if self.cache:
                cache_key = f"embeddings:{embedding_data.person_id}"
                await self.cache.delete(cache_key)

            return embedding
        except Exception as e:
            await self.session.rollback()
            logger.error("Error creating face embedding", error=str(e))
            raise

    async def get_person_embeddings(
        self, person_id: uuid.UUID, active_only: bool = True
    ) -> List[FaceEmbedding]:
        """Get all face embeddings for a person."""
        try:
            # Try cache first
            if self.cache:
                cache_key = f"embeddings:{person_id}"
                cached_data = await self.cache.get(cache_key)
                if cached_data:
                    return cached_data

            # Query database
            stmt = (
                select(FaceEmbedding)
                .where(
                    and_(
                        FaceEmbedding.person_id == person_id,
                        FaceEmbedding.is_active == True if active_only else True,
                    )
                )
                .order_by(FaceEmbedding.created_at.desc())
            )
            result = await self.session.execute(stmt)
            embeddings = result.scalars().all()

            # Cache results
            if self.cache and embeddings:
                await self.cache.set(cache_key, embeddings, ttl=3600)

            return embeddings
        except Exception as e:
            logger.error("Error getting person embeddings", error=str(e))
            raise

    async def find_similar_embeddings(
        self, embedding: List[float], threshold: float = 0.6, limit: int = 5
    ) -> List[Tuple[FaceEmbedding, float]]:
        """
        Find similar embeddings using cosine similarity.
        Returns list of (embedding, similarity_score) tuples.
        """
        try:
            # Convert query embedding to numpy array
            query_embedding = np.array(embedding)

            # Get all active embeddings
            stmt = select(FaceEmbedding).where(FaceEmbedding.is_active == True)
            result = await self.session.execute(stmt)
            embeddings = result.scalars().all()

            # Calculate similarities
            similarities = []
            for emb in embeddings:
                stored_embedding = np.array(emb.embedding)
                similarity = np.dot(query_embedding, stored_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                )
                if similarity >= threshold:
                    similarities.append((emb, float(similarity)))

            # Sort by similarity and return top matches
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

        except Exception as e:
            logger.error("Error finding similar embeddings", error=str(e))
            raise

    async def deactivate_embedding(self, embedding_id: uuid.UUID) -> bool:
        """Deactivate a face embedding."""
        try:
            stmt = select(FaceEmbedding).where(FaceEmbedding.id == embedding_id)
            result = await self.session.execute(stmt)
            embedding = result.scalar_one_or_none()

            if embedding:
                embedding.is_active = False
                await self.session.commit()

                # Invalidate cache
                if self.cache:
                    cache_key = f"embeddings:{embedding.person_id}"
                    await self.cache.delete(cache_key)

                return True
            return False

        except Exception as e:
            await self.session.rollback()
            logger.error("Error deactivating embedding", error=str(e))
            raise

    async def get_embedding_stats(self) -> Dict:
        """Get statistics about stored embeddings."""
        try:
            # Total embeddings
            total_stmt = select(func.count(FaceEmbedding.id))
            total_result = await self.session.execute(total_stmt)
            total_embeddings = total_result.scalar()

            # Active embeddings
            active_stmt = select(func.count(FaceEmbedding.id)).where(
                FaceEmbedding.is_active == True
            )
            active_result = await self.session.execute(active_stmt)
            active_embeddings = active_result.scalar()

            # Total persons
            persons_stmt = select(func.count(Person.id)).where(Person.is_active == True)
            persons_result = await self.session.execute(persons_stmt)
            total_persons = persons_result.scalar()

            # Average quality score
            quality_stmt = select(func.avg(FaceEmbedding.quality_score)).where(
                FaceEmbedding.is_active == True
            )
            quality_result = await self.session.execute(quality_stmt)
            avg_quality = quality_result.scalar() or 0.0

            return {
                "total_embeddings": total_embeddings,
                "active_embeddings": active_embeddings,
                "total_persons": total_persons,
                "average_quality_score": float(avg_quality),
                "timestamp": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error("Error getting embedding stats", error=str(e))
            raise
