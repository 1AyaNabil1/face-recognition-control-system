from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp
import time
import uuid
import structlog
from typing import Callable, Optional
import sentry_sdk
from prometheus_client import Counter, Histogram
from app.core.monitoring import REQUEST_COUNT, REQUEST_LATENCY
from app.core.security import SecurityError
import json
import os

# Initialize logger
logger = structlog.get_logger(__name__)

# Middleware configuration
TRUSTED_HOSTS = os.getenv("TRUSTED_HOSTS", "localhost").split(",")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
REQUEST_ID_HEADER = "X-Request-ID"
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "10485760"))  # 10MB


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context to logs."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate request ID
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())

        # Add context to structlog
        logger.bind(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=request.client.host,
        )

        # Add context to Sentry
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("request_id", request_id)
            scope.set_tag("client_ip", request.client.host)

        # Add request ID to response headers
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()

        response = await call_next(request)

        # Record request duration
        duration = time.time() - start_time
        REQUEST_LATENCY.labels(
            method=request.method, endpoint=request.url.path
        ).observe(duration)

        # Record request count
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()

        return response


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security checks."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Check request size
        if request.headers.get("content-length"):
            content_length = int(request.headers["content-length"])
            if content_length > MAX_REQUEST_SIZE:
                raise SecurityError(
                    f"Request size exceeds maximum allowed size of {MAX_REQUEST_SIZE / 1024 / 1024:.1f}MB"
                )

        # Check host header
        host = request.headers.get("host", "").split(":")[0]
        if host not in TRUSTED_HOSTS and "*" not in TRUSTED_HOSTS:
            raise SecurityError(f"Invalid host header: {host}")

        return await call_next(request)


class CORSMiddleware(BaseHTTPMiddleware):
    """Custom CORS middleware."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        if request.method == "OPTIONS":
            response = Response()
        else:
            response = await call_next(request)

        origin = request.headers.get("origin")
        if origin and (origin in CORS_ORIGINS or "*" in CORS_ORIGINS):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = (
                "GET, POST, PUT, DELETE, OPTIONS"
            )
            response.headers["Access-Control-Allow-Headers"] = (
                "Authorization, Content-Type"
            )
            response.headers["Access-Control-Max-Age"] = "3600"

        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for consistent error handling."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            return await call_next(request)

        except Exception as e:
            # Log error
            logger.error(
                "Request error",
                error=str(e),
                error_type=type(e).__name__,
                path=request.url.path,
                method=request.method,
            )

            # Send to Sentry
            sentry_sdk.capture_exception(e)

            # Create error response
            status_code = 500
            if isinstance(e, SecurityError):
                status_code = 403

            error_response = {
                "success": False,
                "error": {"code": type(e).__name__, "message": str(e)},
            }

            return Response(
                content=json.dumps(error_response),
                status_code=status_code,
                media_type="application/json",
            )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
            client_ip=request.client.host,
            user_agent=request.headers.get("user-agent"),
        )

        response = await call_next(request)

        # Log response
        duration = time.time() - start_time
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
        )

        return response


def setup_middleware(app: ASGIApp):
    """Set up all middleware for the application."""
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(CORSMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
