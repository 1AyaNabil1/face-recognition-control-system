from prometheus_client import Counter, Histogram, Gauge, Info
import time
from typing import Callable
from functools import wraps
import psutil
import torch
from fastapi import Request
import structlog
from opentelemetry import trace
from opentelemetry.trace.status import Status, StatusCode

# Initialize logger
logger = structlog.get_logger(__name__)

# Request metrics
REQUEST_LATENCY = Histogram(
    "face_recognition_request_latency_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
)

REQUEST_COUNT = Counter(
    "face_recognition_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)

# Model metrics
MODEL_INFERENCE_TIME = Histogram(
    "face_recognition_model_inference_seconds",
    "Model inference time in seconds",
    ["model_name", "operation"],
)

MODEL_ERROR_COUNT = Counter(
    "face_recognition_model_errors_total",
    "Total number of model errors",
    ["model_name", "error_type"],
)

# System metrics
SYSTEM_MEMORY = Gauge(
    "face_recognition_system_memory_bytes", "System memory usage in bytes", ["type"]
)

GPU_MEMORY = Gauge(
    "face_recognition_gpu_memory_bytes", "GPU memory usage in bytes", ["device"]
)

CPU_USAGE = Gauge("face_recognition_cpu_usage_percent", "CPU usage percentage")

# Database metrics
DB_CONNECTION_COUNT = Gauge(
    "face_recognition_db_connections", "Number of active database connections"
)

DB_QUERY_LATENCY = Histogram(
    "face_recognition_db_query_latency_seconds",
    "Database query latency in seconds",
    ["operation"],
)

# Cache metrics
CACHE_HIT_COUNT = Counter(
    "face_recognition_cache_hits_total", "Total number of cache hits"
)

CACHE_MISS_COUNT = Counter(
    "face_recognition_cache_misses_total", "Total number of cache misses"
)

# System info
SYSTEM_INFO = Info("face_recognition_system", "System information")


def init_system_info():
    """Initialize system information metrics."""
    SYSTEM_INFO.info(
        {
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": str(torch.cuda.is_available()),
            "gpu_count": str(torch.cuda.device_count())
            if torch.cuda.is_available()
            else "0",
            "cpu_count": str(psutil.cpu_count()),
        }
    )


def update_system_metrics():
    """Update system metrics."""
    # Memory metrics
    memory = psutil.virtual_memory()
    SYSTEM_MEMORY.labels(type="total").set(memory.total)
    SYSTEM_MEMORY.labels(type="used").set(memory.used)
    SYSTEM_MEMORY.labels(type="available").set(memory.available)

    # CPU metrics
    CPU_USAGE.set(psutil.cpu_percent())

    # GPU metrics if available
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory = torch.cuda.memory_stats(i)
            GPU_MEMORY.labels(device=f"cuda:{i}").set(
                memory["allocated_bytes.all.current"]
            )


class MetricsMiddleware:
    """Middleware for collecting request metrics."""

    async def __call__(self, request: Request, call_next: Callable):
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


def track_model_inference(model_name: str, operation: str):
    """Decorator to track model inference metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Record inference time
                MODEL_INFERENCE_TIME.labels(
                    model_name=model_name, operation=operation
                ).observe(duration)

                return result

            except Exception as e:
                # Record error
                MODEL_ERROR_COUNT.labels(
                    model_name=model_name, error_type=type(e).__name__
                ).inc()

                # Set trace status
                span = trace.get_current_span()
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(e)

                raise

        return wrapper

    return decorator


def track_database_query(operation: str):
    """Decorator to track database query metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                # Record query latency
                DB_QUERY_LATENCY.labels(operation=operation).observe(duration)

                return result

            except Exception as e:
                logger.error(
                    "Database error",
                    operation=operation,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

        return wrapper

    return decorator


def track_cache_operation(func):
    """Decorator to track cache operations."""

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)

            if result is not None:
                CACHE_HIT_COUNT.inc()
            else:
                CACHE_MISS_COUNT.inc()

            return result

        except Exception as e:
            logger.error("Cache error", error=str(e), error_type=type(e).__name__)
            raise

    return wrapper


# Initialize metrics
init_system_info()
