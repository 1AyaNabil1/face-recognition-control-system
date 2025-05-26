from pydantic_settings import BaseSettings
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import json
import structlog
from functools import lru_cache

# Initialize logger
logger = structlog.get_logger(__name__)


class APISettings(BaseSettings):
    """API configuration settings."""

    VERSION: str = "v1"
    TITLE: str = "Face Recognition API"
    DESCRIPTION: str = "Production-grade face recognition system"
    DOCS_URL: str = "/api/v1/docs"
    REDOC_URL: str = "/api/v1/redoc"
    OPENAPI_URL: str = "/api/v1/openapi.json"
    ROOT_PATH: str = ""


class SecuritySettings(BaseSettings):
    """Security configuration settings."""

    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    CORS_ORIGINS: List[str] = ["*"]
    TRUSTED_HOSTS: List[str] = ["localhost"]
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png"]
    RATE_LIMIT_PER_MINUTE: int = 100

    class Config:
        env_file = ".env"


class ModelSettings(BaseSettings):
    """Model configuration settings."""

    MODEL_PATH: Path = Path("/root/models")
    CONFIDENCE_THRESHOLD: float = 0.6
    MIN_FACE_SIZE: int = 30
    MAX_FACES: int = 20
    USE_GPU: bool = True
    GPU_MEMORY_FRACTION: float = 0.8
    ENABLE_AGE_GENDER: bool = True
    ENABLE_EMOTION: bool = True
    ENABLE_ANTI_SPOOFING: bool = True
    ENABLE_QUALITY_CHECK: bool = True

    class Config:
        env_file = ".env"


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    DATABASE_URL: str
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SSL: bool = False

    class Config:
        env_file = ".env"


class MonitoringSettings(BaseSettings):
    """Monitoring configuration settings."""

    SENTRY_DSN: Optional[str] = None
    SENTRY_ENVIRONMENT: str = "production"
    SENTRY_TRACES_SAMPLE_RATE: float = 0.2
    PROMETHEUS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: str = "/var/log/face-recognition/app.log"
    LOG_ROTATION: str = "1 day"
    LOG_RETENTION: str = "30 days"

    class Config:
        env_file = ".env"


class StorageSettings(BaseSettings):
    """Storage configuration settings."""

    S3_BUCKET_NAME: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_REGION: str = "us-east-1"
    S3_ENDPOINT: Optional[str] = None
    LOCAL_STORAGE_PATH: Path = Path("/app/data")

    class Config:
        env_file = ".env"


class Settings(BaseSettings):
    """Main configuration settings."""

    # Environment
    ENV: str = "production"
    DEBUG: bool = False
    TESTING: bool = False

    # Component settings
    api: APISettings = APISettings()
    security: SecuritySettings = SecuritySettings()
    model: ModelSettings = ModelSettings()
    database: DatabaseSettings = DatabaseSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    storage: StorageSettings = StorageSettings()

    # Feature flags
    FEATURE_FLAGS: Dict[str, bool] = {
        "enable_age_gender": True,
        "enable_emotion": True,
        "enable_anti_spoofing": True,
        "enable_quality_check": True,
        "enable_caching": True,
    }

    class Config:
        env_file = ".env"

    def update_feature_flags(self, flags: Dict[str, bool]):
        """Update feature flags."""
        self.FEATURE_FLAGS.update(flags)
        logger.info("Feature flags updated", flags=flags)

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "env": self.ENV,
            "debug": self.DEBUG,
            "testing": self.TESTING,
            "api": self.api.dict(),
            "security": self.security.dict(exclude={"JWT_SECRET_KEY"}),
            "model": self.model.dict(),
            "database": self.database.dict(exclude={"DATABASE_URL", "REDIS_PASSWORD"}),
            "monitoring": self.monitoring.dict(exclude={"SENTRY_DSN"}),
            "storage": self.storage.dict(exclude={"S3_ACCESS_KEY", "S3_SECRET_KEY"}),
            "feature_flags": self.FEATURE_FLAGS,
        }

    def save_to_file(self, path: Path):
        """Save settings to file."""
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info("Settings saved to file", path=str(path))
        except Exception as e:
            logger.error("Failed to save settings", error=str(e))
            raise


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Load settings
settings = get_settings()


# Validate required settings
def validate_settings():
    """Validate required configuration settings."""
    required_settings = [
        (settings.security.JWT_SECRET_KEY, "JWT_SECRET_KEY"),
        (settings.database.DATABASE_URL, "DATABASE_URL"),
    ]

    missing_settings = []
    for value, name in required_settings:
        if not value:
            missing_settings.append(name)

    if missing_settings:
        raise ValueError(
            f"Missing required configuration settings: {', '.join(missing_settings)}"
        )

    logger.info("Configuration settings validated successfully")
