from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.detection.face_detector import extract_face
from models.vggface2_model import get_embedding
from app.database.db_manager import insert_embedding
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognizer:
    def __init__(self, embedder, database, threshold=0.6, min_quality=0.5):
        self.embedder = embedder
        self.database = database
        self.threshold = threshold
        self.min_quality = min_quality

    def _get_mean_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Get mean embeddings for each person with validation."""
        records = self.database.fetch_all_embeddings()
        grouped = defaultdict(list)

        # Group embeddings by name
        for name, emb in records:
            if isinstance(emb, list) and len(emb) == 512:
                grouped[name].append(np.array(emb))

        cleaned = []
        for name, embs in grouped.items():
            try:
                # Convert to numpy array and validate shape
                arr = np.array(embs)
                if len(arr.shape) == 2 and arr.shape[1] == 512:
                    # Calculate mean embedding
                    mean = np.mean(arr, axis=0)
                    # Normalize the mean embedding
                    mean = mean / np.linalg.norm(mean)
                    cleaned.append((name, mean))
                else:
                    logger.warning(f"Invalid embedding shape for {name}: {arr.shape}")
            except Exception as e:
                logger.error(f"Error processing embeddings for {name}: {e}")

        return cleaned

    def recognize(self, face: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Recognize a face with quality check and smart matching."""
        # Get embedding with quality check
        embedding, quality = self.embedder.get_embedding(face)

        if embedding is None or quality < self.min_quality:
            logger.warning(f"Face quality too low: {quality:.2f}")
            return "Unknown", 0.0, []

        # Normalize the embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Get mean embeddings from database
        mean_embeddings = self._get_mean_embeddings()
        if len(mean_embeddings) == 0:
            logger.warning("No valid embeddings in database")
            return "Unknown", 0.0, []

        # Calculate similarity scores
        scores = []
        for label, stored_emb in mean_embeddings:
            score = cosine_similarity([embedding], [stored_emb])[0][0]
            scores.append((label, score))

        # Sort by similarity score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Smart matching logic
        if len(scores) >= 2:
            best_score = scores[0][1]
            second_best = scores[1][1]

            # If the difference between top 2 scores is small, require higher threshold
            if (best_score - second_best) < 0.1:
                effective_threshold = self.threshold + 0.1
            else:
                effective_threshold = self.threshold
        else:
            effective_threshold = self.threshold

        # Log top matches for debugging
        logger.info("\n[📊 Similarity Scores]")
        for label, score in scores[:3]:
            logger.info(f"  - {label}: {score:.4f}")

        best_label, best_score = scores[0]
        if best_score >= effective_threshold:
            return best_label, best_score, scores[:3]

        return "Unknown", best_score, scores[:3]

    def add_new_person(self, image: np.ndarray, name: str) -> bool:
        """Add a new person with quality checks."""
        # Get embedding with quality check
        embedding, quality = self.embedder.get_embedding(image)

        if embedding is None or quality < self.min_quality:
            logger.error(f"Cannot add person - face quality too low: {quality:.2f}")
            return False

        try:
            self.database.insert_embedding(name, embedding.tolist(), None)
            logger.info(f"Added new person '{name}' to database")
            return True
        except Exception as e:
            logger.error(f"Error adding person to database: {e}")
            return False
