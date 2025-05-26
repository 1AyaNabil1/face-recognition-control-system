# app/detection/face_detector.py
import cv2
import numpy as np
from typing import Optional, Tuple, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class FaceDetector:
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not Path(cascade_path).exists():
            raise FileNotFoundError(f"Haar cascade file not found at {cascade_path}")
        self.cascade = cv2.CascadeClassifier(cascade_path)


def extract_face(
    image: Union[str, np.ndarray],
    bbox: Optional[Tuple[int, int, int, int]] = None,
    target_size: Tuple[int, int] = (224, 224),
) -> Optional[np.ndarray]:
    """
    Extract and preprocess face from image using bounding box or face detection.

    Args:
        image: Image path or numpy array
        bbox: Optional bounding box (x1, y1, x2, y2). If None, will detect face.
        target_size: Output size for the face image

    Returns:
        Preprocessed face image or None if no face found
    """
    try:
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                logger.error(f"Could not load image from {image}")
                return None
        else:
            img = image.copy()

        # Convert to RGB if needed
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:  # RGBA
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Detect face if bbox not provided
        if bbox is None:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            faces = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            ).detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(faces) == 0:
                logger.warning("No face detected in image")
                return None
            x, y, w, h = faces[0]
            bbox = (x, y, x + w, y + h)

        # Extract face using bbox
        x1, y1, x2, y2 = bbox
        if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
            logger.warning("Face bounding box outside image boundaries")
            return None

        face = img[y1:y2, x1:x2]

        # Basic face validation
        if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
            logger.warning("Extracted face has invalid dimensions")
            return None

        # Resize to target size
        face_resized = cv2.resize(face, target_size)

        return face_resized

    except Exception as e:
        logger.error(f"Error in extract_face: {str(e)}")
        return None


def detect_faces(image: Union[str, np.ndarray]) -> list:
    """
    Detect all faces in an image using Haar Cascade.

    Args:
        image: Image path or numpy array

    Returns:
        List of face bounding boxes (x1, y1, x2, y2)
    """
    try:
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                logger.error(f"Could not load image from {image}")
                return []
        else:
            img = image.copy()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Convert to bounding boxes
        boxes = []
        for x, y, w, h in faces:
            boxes.append((x, y, x + w, y + h))

        return boxes

    except Exception as e:
        logger.error(f"Error in detect_faces: {str(e)}")
        return []


def validate_face_quality(
    face_image: np.ndarray,
    min_size: int = 30,
    min_brightness: float = 40,
    max_brightness: float = 250,
    min_contrast: float = 20,
    min_sharpness: float = 100,
) -> Tuple[bool, float]:
    """
    Assess face image quality for recognition.

    Args:
        face_image: Face image array
        min_size: Minimum face dimension
        min_brightness: Minimum average pixel value
        max_brightness: Maximum average pixel value
        min_contrast: Minimum standard deviation
        min_sharpness: Minimum Laplacian variance

    Returns:
        Tuple of (is_valid, quality_score)
    """
    try:
        if face_image is None or face_image.size == 0:
            return False, 0.0

        # Check size
        h, w = face_image.shape[:2]
        if h < min_size or w < min_size:
            return False, 0.0

        # Check brightness
        brightness = np.mean(face_image)
        if brightness < min_brightness or brightness > max_brightness:
            return False, 0.0

        # Check contrast
        contrast = np.std(face_image)
        if contrast < min_contrast:
            return False, 0.0

        # Check sharpness
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        if sharpness < min_sharpness:
            return False, 0.0

        # Calculate quality score (0-1)
        size_score = min(1.0, min(h, w) / 224)  # Normalize to common size
        brightness_score = 1.0 - abs(brightness - 128) / 128  # Optimal around 128
        contrast_score = min(1.0, contrast / 100)
        sharpness_score = min(1.0, sharpness / 1000)

        # Weighted average of scores
        quality_score = 0.25 * (
            size_score + brightness_score + contrast_score + sharpness_score
        )

        return True, quality_score

    except Exception as e:
        logger.error(f"Error in validate_face_quality: {str(e)}")
        return False, 0.0
