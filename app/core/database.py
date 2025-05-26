from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy import MetaData, Column, Integer, String, Float, DateTime, Index
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from datetime import datetime
import os
import structlog
from contextlib import asynccontextmanager
from typing import AsyncGenerator
import alembic.config
import alembic.command
from app.core.monitoring import track_database_query

# Initialize logger
logger = structlog.get_logger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+asyncpg://user:password@localhost/face_recognition"
)
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "20"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))

# Create async engine with connection pooling
engine = create_async_engine(
    DATABASE_URL,
    poolclass=AsyncAdaptedQueuePool,
    pool_size=DB_POOL_SIZE,
    max_overflow=DB_MAX_OVERFLOW,
    pool_timeout=DB_POOL_TIMEOUT,
    pool_pre_ping=True,  # Enable connection health checks
    echo=False,  # Set to True for SQL query logging
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create base class for declarative models
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)
Base = declarative_base(metadata=metadata)


@asynccontextmanager
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Get database session with automatic cleanup.

    Usage:
        async with get_db() as db:
            result = await db.execute(query)
    """
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception as e:
        logger.error("Database session error", error=str(e))
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db():
    """Initialize database with required tables."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        logger.info("Database initialized successfully")

    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        raise


async def run_migrations():
    """Run database migrations using Alembic."""
    try:
        config = alembic.config.Config("alembic.ini")
        alembic.command.upgrade(config, "head")

        logger.info("Database migrations completed successfully")

    except Exception as e:
        logger.error("Database migration failed", error=str(e))
        raise


class DatabaseManager:
    """Manager class for database operations."""

    @staticmethod
    @track_database_query(operation="select")
    async def fetch_one(session: AsyncSession, query):
        """Execute query and fetch one result."""
        try:
            result = await session.execute(query)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error("Database fetch_one error", error=str(e))
            raise

    @staticmethod
    @track_database_query(operation="select")
    async def fetch_all(session: AsyncSession, query):
        """Execute query and fetch all results."""
        try:
            result = await session.execute(query)
            return result.scalars().all()
        except Exception as e:
            logger.error("Database fetch_all error", error=str(e))
            raise

    @staticmethod
    @track_database_query(operation="insert")
    async def insert(session: AsyncSession, model):
        """Insert a model instance."""
        try:
            session.add(model)
            await session.commit()
            await session.refresh(model)
            return model
        except Exception as e:
            await session.rollback()
            logger.error("Database insert error", error=str(e))
            raise

    @staticmethod
    @track_database_query(operation="update")
    async def update(session: AsyncSession, model):
        """Update a model instance."""
        try:
            await session.commit()
            await session.refresh(model)
            return model
        except Exception as e:
            await session.rollback()
            logger.error("Database update error", error=str(e))
            raise

    @staticmethod
    @track_database_query(operation="delete")
    async def delete(session: AsyncSession, model):
        """Delete a model instance."""
        try:
            await session.delete(model)
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database delete error", error=str(e))
            raise


# Database models
class FaceEmbedding(Base):
    """Model for storing face embeddings."""

    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    embedding = Column(ARRAY(Float))
    model_name = Column(String)
    confidence = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (Index("ix_face_embeddings_name_model", "name", "model_name"),)


class FaceRecognitionLog(Base):
    """Model for logging face recognition attempts."""

    __tablename__ = "face_recognition_logs"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True)
    image_hash = Column(String)
    model_name = Column(String)
    matches = Column(JSONB)
    inference_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (Index("ix_face_recognition_logs_created_at", "created_at"),)
