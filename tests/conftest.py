"""Shared fixtures.

The real pipeline needs YOLOv8 weights and InsightFace models; these tests
replace both with small deterministic fakes so they run offline in seconds.
"""

import base64

import cv2
import numpy as np
import pytest

from api.app import create_app
from api.recognition_service import RecognitionService
from app.database.db_manager import EmbeddingDatabase
from app.settings import Settings

DIM = 512


def unit_vector(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=DIM)
    return v / np.linalg.norm(v)


def solid_image(bgr, size=(120, 160)) -> np.ndarray:
    """A solid-colour BGR image; the fake embedder keys identities off the colour."""
    image = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    image[:] = bgr
    return image


def to_b64(image: np.ndarray, ext: str = ".png") -> str:
    ok, buffer = cv2.imencode(ext, image)
    assert ok
    return base64.b64encode(buffer.tobytes()).decode("ascii")


class FakeDetector:
    """Stands in for YOLOFaceDetector: returns fixed boxes."""

    def __init__(self, boxes=((20, 20, 100, 100),)):
        self.boxes = list(boxes)

    def detect_faces(self, image):
        return list(self.boxes)

    def draw_boxes(self, image, boxes):
        return image


class FakeEmbedder:
    """Stands in for FaceEmbedder: the embedding depends only on the crop colour.

    Crops of the same solid colour get the same identity vector, so
    "same colour" behaves like "same person".
    """

    def __init__(self, quality=0.9, reject_reason=None):
        self.quality = quality
        self.reject_reason = reject_reason

    def check_quality(self, face):
        return self.reject_reason

    def get_embedding(self, face):
        if self.reject_reason is not None:
            return None, 0.0
        b, g, r = (int(c) for c in face.reshape(-1, 3).mean(axis=0).round())
        return unit_vector(b * 65536 + g * 256 + r), self.quality


@pytest.fixture
def settings(tmp_path):
    return Settings(
        db_path=str(tmp_path / "embeddings.db"),
        recognition_log_path=str(tmp_path / "logs" / "recognition_log.csv"),
        max_image_bytes=1024 * 1024,
    )


@pytest.fixture
def database(settings):
    return EmbeddingDatabase(
        db_path=settings.db_path, log_path=settings.recognition_log_path
    )


@pytest.fixture
def make_service(settings, database):
    def _make(detector=None, embedder=None, service_settings=None):
        return RecognitionService(
            detector=detector or FakeDetector(),
            embedder=embedder or FakeEmbedder(),
            database=database,
            settings=service_settings or settings,
        )

    return _make


@pytest.fixture
def make_client(settings, make_service):
    def _make(app_settings=None, service=None, **service_kwargs):
        app_settings = app_settings or settings
        service = service or make_service(
            service_settings=app_settings, **service_kwargs
        )
        app = create_app(settings=app_settings, service=service)
        app.config["TESTING"] = True
        return app.test_client()

    return _make


@pytest.fixture
def client(make_client):
    return make_client()
