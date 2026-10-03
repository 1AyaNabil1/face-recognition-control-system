"""Runtime settings for the YOLOv8 + InsightFace recognition pipeline.

Every value can be overridden with an environment variable (see .env.example),
so nothing machine-specific has to be edited in the code.
"""

import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _env_list(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "")
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    # Storage
    db_path: str = "app/database/embeddings.db"
    recognition_log_path: str = "logs/recognition_log.csv"
    log_recognitions: bool = True

    # Models
    yolo_model_path: str = "app/models/yolo/yolov8n-face-lindevs.pt"
    yolo_confidence: float = 0.2
    insightface_model: str = "buffalo_l"
    insightface_root: str = "~/.insightface"
    insightface_det_size: int = 640

    # Matching
    match_threshold: float = 0.6
    min_face_quality: float = 0.5

    # HTTP API
    max_image_bytes: int = 10 * 1024 * 1024
    api_key: str = ""
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        defaults = cls()
        return cls(
            db_path=_env_str("FRCS_DB_PATH", defaults.db_path),
            recognition_log_path=_env_str(
                "FRCS_RECOGNITION_LOG_PATH", defaults.recognition_log_path
            ),
            log_recognitions=_env_bool(
                "FRCS_LOG_RECOGNITIONS", defaults.log_recognitions
            ),
            yolo_model_path=_env_str("FRCS_YOLO_MODEL_PATH", defaults.yolo_model_path),
            yolo_confidence=_env_float(
                "FRCS_YOLO_CONFIDENCE", defaults.yolo_confidence
            ),
            insightface_model=_env_str(
                "FRCS_INSIGHTFACE_MODEL", defaults.insightface_model
            ),
            insightface_root=_env_str(
                "FRCS_INSIGHTFACE_ROOT", defaults.insightface_root
            ),
            insightface_det_size=_env_int(
                "FRCS_INSIGHTFACE_DET_SIZE", defaults.insightface_det_size
            ),
            match_threshold=_env_float(
                "FRCS_MATCH_THRESHOLD", defaults.match_threshold
            ),
            min_face_quality=_env_float(
                "FRCS_MIN_FACE_QUALITY", defaults.min_face_quality
            ),
            max_image_bytes=_env_int("FRCS_MAX_IMAGE_BYTES", defaults.max_image_bytes),
            api_key=_env_str("FRCS_API_KEY", defaults.api_key),
            cors_origins=_env_list("FRCS_CORS_ORIGINS"),
        )


def load_dotenv_if_available() -> None:
    """Load a local .env file when python-dotenv is installed (optional)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
