import numpy as np
import cv2
from typing import Optional, Tuple
import logging
from tensorflow.keras.models import Model
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

logger = logging.getLogger(__name__)


class VGGFace2Embedder:
    def __init__(self, model_name: str = "resnet50"):
        """
        Initialize VGGFace2 model for embedding extraction.

        Args:
            model_name: Either 'resnet50' or 'vgg16'
        """
        try:
            # Load base model
            base_model = VGGFace(
                model=model_name,
                include_top=False,
                input_shape=(224, 224, 3),
                pooling="avg",
            )

            # Create embedding model
            self.model = Model(
                inputs=base_model.input, outputs=base_model.get_layer("avg_pool").output
            )

            self.input_shape = (224, 224)
            logger.info(f"Loaded VGGFace2 {model_name} model successfully")

        except Exception as e:
            logger.error(f"Error loading VGGFace2 model: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for VGGFace2 model.

        Args:
            image: RGB image array

        Returns:
            Preprocessed image array
        """
        try:
            # Resize if needed
            if image.shape[:2] != self.input_shape:
                image = cv2.resize(image, self.input_shape)

            # Ensure RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            # Convert to float32
            image = image.astype("float32")

            # Expand dims for batch
            image = np.expand_dims(image, axis=0)

            # Preprocess using VGGFace2 requirements
            image = preprocess_input(image, version=2)  # version=2 for ResNet

            return image

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def get_embedding(
        self, face_image: np.ndarray
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract embedding from face image.

        Args:
            face_image: RGB face image array

        Returns:
            Tuple of (embedding array, quality score)
        """
        try:
            if face_image is None or face_image.size == 0:
                logger.warning("Invalid input image")
                return None, 0.0

            # Preprocess image
            preprocessed = self.preprocess_image(face_image)
            if preprocessed is None:
                return None, 0.0

            # Extract embedding
            embedding = self.model.predict(preprocessed, verbose=0)[0]

            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm == 0:
                logger.warning("Zero embedding detected")
                return None, 0.0

            normalized_embedding = embedding / norm

            # Calculate basic quality score
            quality = min(1.0, norm / 100)  # Normalize embedding magnitude

            return normalized_embedding, quality

        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            return None, 0.0

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1, emb2: Normalized embedding arrays

        Returns:
            Similarity score between 0 and 1
        """
        try:
            if emb1 is None or emb2 is None:
                return 0.0

            similarity = np.dot(emb1, emb2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
