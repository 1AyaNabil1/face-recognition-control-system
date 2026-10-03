import logging
import os
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]


def expand_box(box: Box, image_shape: Tuple[int, ...], margin: float = 0.3) -> Box:
    """Grow a (x1, y1, x2, y2) box by `margin` of its size, clipped to the image."""
    x1, y1, x2, y2 = box
    h, w = image_shape[:2]
    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)
    return (max(0, x1 - dx), max(0, y1 - dy), min(w, x2 + dx), min(h, y2 + dy))


class YOLOFaceDetector:
    def __init__(
        self,
        model_path="app/models/yolo/yolov8n-face-lindevs.pt",
        confidence=0.35,  # Increased default confidence
        iou_threshold=0.5,
        min_face_size=30,
        model=None,
    ):
        if model is None:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"YOLO model not found at {model_path}")
            # Imported lazily so the module works without ultralytics installed.
            from ultralytics import YOLO

            model = YOLO(model_path)

        self.model = model
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.min_face_size = min_face_size
        self.fallback_cascade = None
        if hasattr(cv2, "CascadeClassifier"):
            self.fallback_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        else:
            # OpenCV 5 dropped the Haar cascade API from the main package
            logger.warning("cv2.CascadeClassifier unavailable; Haar fallback disabled")

    def _validate_face_box(self, box: Box, image_shape: Tuple[int, ...]) -> bool:
        """Validate face bounding box dimensions and position."""
        x1, y1, x2, y2 = box
        h, w = image_shape[:2]

        # Check if box is within image bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False

        # Check minimum face size
        face_w, face_h = x2 - x1, y2 - y1
        if face_w < self.min_face_size or face_h < self.min_face_size:
            return False

        # Check aspect ratio (should be roughly square)
        aspect_ratio = face_w / face_h
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:
            return False

        return True

    def detect_faces(self, frame: np.ndarray) -> List[Box]:
        """Detect faces with validation and smart fallback."""
        if frame is None or frame.size == 0:
            logger.warning("Invalid input frame")
            return []

        # YOLO detection
        results = self.model.predict(
            source=frame, conf=self.confidence, iou=self.iou_threshold, verbose=False
        )

        boxes = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf >= self.confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if self._validate_face_box((x1, y1, x2, y2), frame.shape):
                            boxes.append((x1, y1, x2, y2))

        # Smart fallback to Haar Cascade
        if not boxes and self.fallback_cascade is not None:
            logger.info("YOLO detection failed, trying Haar Cascade")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.fallback_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size),
            )

            for x, y, w, h in faces:
                # Cast numpy ints to plain ints so boxes are JSON-serialisable
                box = (int(x), int(y), int(x + w), int(y + h))
                if self._validate_face_box(box, frame.shape):
                    boxes.append(box)

        return boxes

    def draw_boxes(
        self,
        frame: np.ndarray,
        boxes: List[Box],
        color=(0, 255, 0),
    ) -> np.ndarray:
        """Draw face boxes with optional labels."""
        annotated = frame.copy()
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # Add box number for multiple faces
            if len(boxes) > 1:
                cv2.putText(
                    annotated,
                    f"Face {i + 1}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
        return annotated
