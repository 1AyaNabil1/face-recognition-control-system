"""Modal deployment configuration for the face recognition system."""

import sys
import os
import modal

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Initialize Modal app
app = modal.App("face-recognition-system")

# Create image with required dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(
        "libgl1-mesa-glx",  # OpenCV dependencies
        "libglib2.0-0",
        "postgresql-client",  # PostgreSQL client
    )
    .pip_install(
        # FastAPI and dependencies
        "fastapi[all]==0.109.2",
        "uvicorn==0.27.1",
        "python-multipart==0.0.9",
        # Database
        "sqlalchemy[asyncio]==2.0.27",
        "asyncpg==0.29.0",
        # ML and Vision libraries
        "onnxruntime==1.16.0",
        "insightface==0.7.3",
        "numpy==1.26.4",
        "opencv-python==4.9.0.80",
        "pillow==10.4.0",
        "scikit-learn==1.5.1",
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("core-secrets")],
)
@modal.asgi_app()
def fastapi_app():
    """
    Deploy the FastAPI application with core configuration.
    Includes all endpoints:
    - Main API (/recognize, /register, /verify)
    - Mobile API (/api/v1/mobile/*)
    """
    # The secrets are automatically mounted as environment variables
    # No need to manually set them

    # Import and return the FastAPI app
    from app.main import app

    return app


if __name__ == "__main__":
    modal.runner.main()
