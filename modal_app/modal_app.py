import sys
import os
import modal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

app = modal.App("face-recognition-system")

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


@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    """
    Deploy the FastAPI application.
    Includes all endpoints:
    - Main API (/recognize, /register, /verify)
    - Mobile API (/api/v1/mobile/*)
    - Health checks
    - Monitoring
    """
    from app.main import app

    return app
