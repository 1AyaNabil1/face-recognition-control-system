import logging
import logging.handlers
import json
import os
from datetime import datetime
from pathlib import Path
import structlog
from typing import Any, Dict

# Create logs directory if it doesn't exist
log_dir = Path("/app/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Configure structlog
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)


class CustomJSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add custom fields
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "user_id"):
            log_data["user_id"] = record.user_id

        return json.dumps(log_data)


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "/app/logs/app.log",
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5,
) -> None:
    """
    Set up production-grade logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
    """
    # Set root logger level
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Create formatters
    json_formatter = CustomJSONFormatter()

    # Console handler with color formatting
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(json_formatter)
    root_logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(json_formatter)
    root_logger.addHandler(file_handler)

    # Set specific log levels for third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    # Create logger for the application
    logger = structlog.get_logger()
    logger.info(
        "Logging configured",
        log_level=log_level,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
    )


class RequestIdFilter(logging.Filter):
    """Filter to add request ID to log records."""

    def __init__(self, request_id: str):
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = self.request_id
        return True


class UserIdFilter(logging.Filter):
    """Filter to add user ID to log records."""

    def __init__(self, user_id: str):
        self.user_id = user_id

    def filter(self, record: logging.LogRecord) -> bool:
        record.user_id = self.user_id
        return True


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Get a logger instance with the given name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


def log_request(
    logger: structlog.BoundLogger,
    request_id: str,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:
    """
    Log an API request with relevant information.

    Args:
        logger: Logger instance
        request_id: Unique request ID
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration_ms: Request duration in milliseconds
    """
    logger.info(
        "API request",
        request_id=request_id,
        method=method,
        path=path,
        status_code=status_code,
        duration_ms=duration_ms,
    )


def log_model_inference(
    logger: structlog.BoundLogger,
    model_name: str,
    inference_time_ms: float,
    success: bool,
    error: str = None,
) -> None:
    """
    Log model inference metrics.

    Args:
        logger: Logger instance
        model_name: Name of the model
        inference_time_ms: Inference time in milliseconds
        success: Whether inference was successful
        error: Error message if inference failed
    """
    log_data = {
        "model_name": model_name,
        "inference_time_ms": inference_time_ms,
        "success": success,
    }
    if error:
        log_data["error"] = error

    logger.info("Model inference", **log_data)
