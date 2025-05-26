"""Modal deployment configuration for the face recognition system."""

import sys
import os
import modal
from typing import Dict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Initialize Modal app and stub
app = modal.App("face-recognition-system")
stub = modal.Stub()

# Create image with all required dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(
        "libgl1-mesa-glx",  # OpenCV dependencies
        "libglib2.0-0",
        "postgresql-client",  # PostgreSQL client
        "redis-tools",  # Redis tools
    )
    .pip_install(
        # FastAPI and dependencies
        "fastapi[all]==0.109.2",
        "uvicorn==0.27.1",
        "python-multipart==0.0.9",
        "email-validator==2.1.0.post1",
        # Database and caching
        "sqlalchemy[asyncio]==2.0.27",
        "asyncpg==0.29.0",
        "redis==5.0.1",
        "aioredis==2.0.1",
        # ML and Vision libraries
        "onnxruntime==1.16.0",
        "insightface==0.7.3",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
        "pillow==10.4.0",
        "scikit-learn==1.5.1",
        # Monitoring and logging
        "structlog==24.1.0",
        "prometheus-client==0.19.0",
        "sentry-sdk[fastapi]==1.40.4",
        # Testing
        "pytest==7.4.3",
        "pytest-asyncio==0.21.1",
        "httpx==0.25.2",
    )
)

# Define secrets
database_secret = modal.Secret.from_name("database")
redis_secret = modal.Secret.from_name("redis")
security_secret = modal.Secret.from_name("security")
model_secret = modal.Secret.from_name("model")
monitoring_secret = modal.Secret.from_name("monitoring")
storage_secret = modal.Secret.from_name("storage")
feature_secret = modal.Secret.from_name("features")


def configure_environment(secret_dict: Dict[str, str], prefix: str = ""):
    """Configure environment variables from a secret dictionary."""
    for key, value in secret_dict.items():
        env_key = f"{prefix}{key}" if prefix else key
        os.environ[env_key] = str(value)


@app.function(
    image=image,
    secrets=[
        database_secret,
        redis_secret,
        security_secret,
        model_secret,
        monitoring_secret,
        storage_secret,
        feature_secret,
    ],
    gpu="any",  # Request GPU for model inference
)
@modal.asgi_app()
def fastapi_app():
    """
    Deploy the FastAPI application with secrets injection.
    Includes all endpoints:
    - Main API (/recognize, /register, /verify)
    - Mobile API (/api/v1/mobile/*)
    - Health checks
    - Monitoring
    """
    # Configure environment variables from secrets
    configure_environment(database_secret)  # Database configuration
    configure_environment(redis_secret)  # Redis configuration
    configure_environment(security_secret)  # Security settings
    configure_environment(model_secret)  # Model settings
    configure_environment(monitoring_secret)  # Monitoring configuration
    configure_environment(storage_secret)  # Storage settings
    configure_environment(feature_secret)  # Feature flags

    # Additional Modal-specific environment variables
    os.environ["ENV"] = "production"
    os.environ["MODAL_ENVIRONMENT"] = "true"

    # Import and return the FastAPI app
    from app.main import app

    return app


if __name__ == "__main__":
    modal.runner.main()
