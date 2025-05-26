import insightface
import cv2
import numpy as np
from typing import Optional, Tuple


class FaceEmbedder:
    def __init__(self, min_face_quality: float = 0.5):
        self.model = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],  # Only load needed modules
        )
        self.model.prepare(ctx_id=0)
        self.min_face_quality = min_face_quality

    def assess_face_quality(self, face_img: np.ndarray) -> float:
        """Assess face image quality based on brightness, contrast and blur."""
        if face_img is None or face_img.size == 0:
            return 0.0

        # Check brightness
        brightness = np.mean(face_img)
        if brightness < 40 or brightness > 250:
            return 0.0

        # Check contrast
        contrast = np.std(face_img)
        if contrast < 20:
            return 0.0

        # Check blur using Laplacian variance
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 100:
            return 0.0

        # Normalize quality score between 0 and 1
        quality = min(1.0, blur_score / 1000)
        return quality

    def get_embedding(self, face: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Get face embedding and quality score."""
        if face is None or face.size == 0:
            return None, 0.0

        # Ensure correct size
        if face.shape[:2] != (112, 112):
            face = cv2.resize(face, (112, 112))

        # Check face quality
        quality = self.assess_face_quality(face)
        if quality < self.min_face_quality:
            return None, quality

        # Get embedding
        try:
            faces = self.model.get(face)
            if faces and len(faces) > 0:
                return faces[0].embedding, quality
        except Exception as e:
            print(f"Error getting embedding: {e}")

        return None, quality
