from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import structlog
from typing import Any, Dict, Optional
import traceback
import sys
from app.core.security import (
    SecurityError,
    InvalidTokenError,
    RateLimitExceededError,
    FileValidationError,
)
import sentry_sdk

# Initialize logger
logger = structlog.get_logger(__name__)


class APIError(Exception):
    """Base class for API errors."""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(message)


class ValidationError(APIError):
    """Raised when request validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class AuthenticationError(APIError):
    """Raised when authentication fails."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR",
        )


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
        )


class ModelError(APIError):
    """Raised when model inference fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="MODEL_ERROR",
            details=details,
        )


async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
    """Handle API errors."""
    error_response = {
        "success": False,
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        },
    }

    # Log error
    logger.error(
        "API error",
        error_code=exc.error_code,
        message=exc.message,
        details=exc.details,
        path=request.url.path,
        method=request.method,
    )

    # Send to Sentry if it's a server error
    if exc.status_code >= 500:
        sentry_sdk.capture_exception(exc)

    return JSONResponse(status_code=exc.status_code, content=error_response)


async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle request validation errors."""
    details = []
    for error in exc.errors():
        details.append(
            {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        )

    error_response = {
        "success": False,
        "error": {
            "code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": details,
        },
    }

    logger.warning(
        "Validation error", path=request.url.path, method=request.method, errors=details
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_response
    )


async def http_error_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Handle HTTP errors."""
    error_response = {
        "success": False,
        "error": {"code": "HTTP_ERROR", "message": str(exc.detail), "details": {}},
    }

    logger.warning(
        "HTTP error",
        status_code=exc.status_code,
        detail=exc.detail,
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(status_code=exc.status_code, content=error_response)


async def security_error_handler(request: Request, exc: SecurityError) -> JSONResponse:
    """Handle security-related errors."""
    if isinstance(exc, InvalidTokenError):
        status_code = status.HTTP_401_UNAUTHORIZED
        error_code = "INVALID_TOKEN"
    elif isinstance(exc, RateLimitExceededError):
        status_code = status.HTTP_429_TOO_MANY_REQUESTS
        error_code = "RATE_LIMIT_EXCEEDED"
    elif isinstance(exc, FileValidationError):
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = "FILE_VALIDATION_ERROR"
    else:
        status_code = status.HTTP_403_FORBIDDEN
        error_code = "SECURITY_ERROR"

    error_response = {
        "success": False,
        "error": {"code": error_code, "message": str(exc), "details": {}},
    }

    logger.warning(
        "Security error",
        error_code=error_code,
        message=str(exc),
        path=request.url.path,
        method=request.method,
    )

    return JSONResponse(status_code=status_code, content=error_response)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unhandled exceptions."""
    # Get full traceback
    exc_info = sys.exc_info()
    traceback_str = "".join(traceback.format_exception(*exc_info))

    error_response = {
        "success": False,
        "error": {
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {"type": type(exc).__name__, "message": str(exc)},
        },
    }

    # Log error with full traceback
    logger.error(
        "Unhandled error",
        error_type=type(exc).__name__,
        error_message=str(exc),
        traceback=traceback_str,
        path=request.url.path,
        method=request.method,
    )

    # Send to Sentry
    sentry_sdk.capture_exception(exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content=error_response
    )


def setup_error_handlers(app):
    """Set up error handlers for the FastAPI application."""
    app.add_exception_handler(APIError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_error_handler)
    app.add_exception_handler(SecurityError, security_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
