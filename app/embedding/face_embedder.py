import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Basic image-statistics gates applied to a face crop before embedding it.
# Sharpness is the variance of the Laplacian on a 112x112 grayscale version of
# the crop, so the value does not depend on the input resolution.
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 250
MIN_CONTRAST = 20
MIN_SHARPNESS = 50

# InsightFace's detector needs some context around the face to find it and
# its landmarks; a tight detector crop is often missed. Each side of the crop
# is padded by this fraction of the crop size before running InsightFace.
CROP_PADDING = 0.5


class FaceEmbedder:
    """512-d ArcFace embeddings from InsightFace for a single face crop.

    The crop is padded and passed through InsightFace's own detector so the
    face is aligned on its 5 landmarks before embedding, which is what the
    ArcFace model was trained on. The returned embedding is L2-normalised and
    the returned quality is InsightFace's detection score (0..1).
    """

    def __init__(
        self,
        min_face_quality: float = 0.5,
        model_name: str = "buffalo_l",
        root: str = "~/.insightface",
        det_size: int = 640,
        analyzer=None,
    ):
        self.min_face_quality = min_face_quality
        if analyzer is None:
            # Imported lazily so the module (and its tests) work without
            # InsightFace installed. Model files are downloaded to `root`
            # on first use.
            from insightface.app import FaceAnalysis

            analyzer = FaceAnalysis(
                name=model_name,
                root=root,
                providers=["CPUExecutionProvider"],
                allowed_modules=["detection", "recognition"],
            )
            analyzer.prepare(ctx_id=0, det_size=(det_size, det_size))
        self.model = analyzer

    @staticmethod
    def check_quality(face_img: np.ndarray) -> Optional[str]:
        """Return a human-readable reason if the crop is unusable, else None."""
        if face_img is None or face_img.size == 0:
            return "empty face crop"
        if face_img.ndim != 3 or face_img.shape[2] != 3:
            return "face crop must be a 3-channel BGR image"

        brightness = float(np.mean(face_img))
        if brightness < MIN_BRIGHTNESS:
            return "image too dark"
        if brightness > MAX_BRIGHTNESS:
            return "image too bright"

        if float(np.std(face_img)) < MIN_CONTRAST:
            return "image contrast too low"

        gray = cv2.cvtColor(cv2.resize(face_img, (112, 112)), cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_SHARPNESS:
            return "image too blurry"

        return None

    @staticmethod
    def _pad(face: np.ndarray) -> np.ndarray:
        pad_y = int(face.shape[0] * CROP_PADDING)
        pad_x = int(face.shape[1] * CROP_PADDING)
        return cv2.copyMakeBorder(
            face, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0
        )

    def get_embedding(self, face: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Embed a BGR face crop.

        Returns (embedding, quality). The embedding is None when the crop
        fails the basic quality checks, InsightFace finds no face in it, or
        the detection score is below `min_face_quality`.
        """
        reason = self.check_quality(face)
        if reason is not None:
            logger.info("Face crop rejected: %s", reason)
            return None, 0.0

        try:
            faces = self.model.get(self._pad(face))
        except Exception:
            logger.exception("InsightFace failed on the face crop")
            return None, 0.0

        if not faces:
            logger.info("InsightFace found no face in the crop")
            return None, 0.0

        # Use the largest face in the crop (the one the detector box was on).
        best = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        quality = float(best.det_score)
        if quality < self.min_face_quality:
            logger.info("Face detection score %.2f below minimum", quality)
            return None, quality

        embedding = np.asarray(best.embedding, dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm == 0.0 or not np.isfinite(norm):
            return None, quality
        return embedding / norm, quality
