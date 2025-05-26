from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import sentry_sdk
from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
from prometheus_client import make_asgi_app
import uvicorn
import structlog
from typing import Dict, Any
import os

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

# Initialize logger
logger = structlog.get_logger(__name__)


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

    # Add health check endpoint
    @app.get("/health")
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint."""
        # Check components
        components = {
            "api": {"status": "healthy"},
            "database": {"status": "unknown"},
            "redis": {"status": "unknown"},
            "model": {"status": "unknown"},
        }

        # TODO: Implement component health checks

        # Return health status
        return {
            "status": "healthy",
            "version": settings.api.VERSION,
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
            # TODO: Initialize database
            # TODO: Initialize Redis
            # TODO: Load models
            pass

        except Exception as e:
            logger.error("Startup failed", error=str(e))
            raise

    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Run shutdown tasks."""
        logger.info("Shutting down application")

        try:
            # TODO: Close database connections
            # TODO: Clear Redis cache
            # TODO: Unload models
            pass

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
