"""Glue between the detector, embedder, recognizer and database.

The HTTP API, the Tkinter GUI and the enrollment script all go through this
class so that enrollment and recognition crop and embed faces the same way.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.database.db_manager import EmbeddingDatabase
from app.detection.yolo_detector import expand_box
from app.recognition.face_recognizer import UNKNOWN, FaceRecognizer
from app.settings import Settings

logger = logging.getLogger(__name__)

# Context kept around the detector box before handing the crop to InsightFace
FACE_CROP_MARGIN = 0.3


@dataclass
class RecognitionResult:
    name: str
    score: float
    top_matches: List[Tuple[str, float]]
    annotated_image: np.ndarray
    face_box: Optional[Tuple[int, int, int, int]] = None
    faces_detected: int = 0


@dataclass
class EnrollmentResult:
    success: bool
    reason: Optional[str] = None
    quality: float = 0.0
    face_box: Optional[Tuple[int, int, int, int]] = None
    faces_detected: int = 0


class RecognitionService:
    def __init__(
        self,
        detector,
        embedder,
        database,
        recognizer: Optional[FaceRecognizer] = None,
        settings: Optional[Settings] = None,
    ):
        self.settings = settings or Settings()
        self.db = database
        self.yolo = detector
        self.embedder = embedder
        self.recognizer = recognizer or FaceRecognizer(
            embedder=embedder,
            database=database,
            threshold=self.settings.match_threshold,
            min_quality=self.settings.min_face_quality,
        )

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "RecognitionService":
        """Build the real pipeline (loads YOLOv8 and InsightFace models)."""
        from app.detection.yolo_detector import YOLOFaceDetector
        from app.embedding.face_embedder import FaceEmbedder

        settings = settings or Settings.from_env()
        return cls(
            detector=YOLOFaceDetector(
                model_path=settings.yolo_model_path,
                confidence=settings.yolo_confidence,
            ),
            embedder=FaceEmbedder(
                min_face_quality=settings.min_face_quality,
                model_name=settings.insightface_model,
                root=settings.insightface_root,
                det_size=settings.insightface_det_size,
            ),
            database=EmbeddingDatabase(
                db_path=settings.db_path, log_path=settings.recognition_log_path
            ),
            settings=settings,
        )

    def _largest_face(self, image: np.ndarray):
        boxes = self.yolo.detect_faces(image)
        if not boxes:
            return None, boxes
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        x1, y1, x2, y2 = expand_box(box, image.shape, FACE_CROP_MARGIN)
        return (box, image[y1:y2, x1:x2]), boxes

    def recognize_image(self, image: np.ndarray) -> RecognitionResult:
        """Recognize the largest face in a BGR image."""
        largest, boxes = self._largest_face(image)
        if largest is None:
            return RecognitionResult(UNKNOWN, 0.0, [], image)

        box, face = largest
        name, score, top_matches = self.recognizer.recognize(face)
        annotated_img = self.yolo.draw_boxes(image.copy(), boxes)

        if name != UNKNOWN and self.settings.log_recognitions:
            self.db.log_recognition_event(name)

        return RecognitionResult(
            name=name,
            score=float(score),
            top_matches=[(label, float(s)) for label, s in top_matches],
            annotated_image=annotated_img,
            face_box=tuple(int(v) for v in box),
            faces_detected=len(boxes),
        )

    def enroll(
        self, image: np.ndarray, name: str, image_path: Optional[str] = None
    ) -> EnrollmentResult:
        """Detect the largest face in a BGR image and store its embedding.

        `image_path` is only recorded for reference; the image itself is not stored.
        """
        largest, boxes = self._largest_face(image)
        if largest is None:
            return EnrollmentResult(False, "no face detected")

        box, face = largest
        face_box = tuple(int(v) for v in box)

        reason = self.embedder.check_quality(face)
        if reason is not None:
            return EnrollmentResult(False, reason, face_box=face_box)

        embedding, quality = self.embedder.get_embedding(face)
        if embedding is None:
            return EnrollmentResult(
                False,
                "face could not be embedded (not found or low detection score)",
                quality=float(quality),
                face_box=face_box,
            )

        if not self.db.insert_embedding(
            name, np.asarray(embedding).tolist(), image_path
        ):
            return EnrollmentResult(
                False, "embedding could not be stored", quality=float(quality)
            )

        return EnrollmentResult(
            True,
            quality=float(quality),
            face_box=face_box,
            faces_detected=len(boxes),
        )
