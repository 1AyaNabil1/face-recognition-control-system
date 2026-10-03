import logging
from collections import defaultdict
from typing import Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 512
UNKNOWN = "Unknown"


def l2_normalize(vector: np.ndarray) -> Optional[np.ndarray]:
    """Return the unit-length version of `vector`, or None if it has no direction."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        return None
    return vector / norm


def mean_embeddings_by_name(
    records: Iterable[Tuple[str, List[float]]],
) -> List[Tuple[str, np.ndarray]]:
    """Build one unit-length template per person from their stored embeddings.

    Each embedding is normalised before averaging so one sample with a large
    norm cannot dominate the template; malformed rows are skipped.
    """
    grouped = defaultdict(list)
    for name, emb in records:
        arr = np.asarray(emb, dtype=np.float64)
        if arr.shape != (EMBEDDING_DIM,):
            logger.warning("Skipping embedding for %s with shape %s", name, arr.shape)
            continue
        unit = l2_normalize(arr)
        if unit is not None:
            grouped[name].append(unit)

    templates = []
    for name, embs in grouped.items():
        template = l2_normalize(np.mean(embs, axis=0))
        if template is not None:
            templates.append((name, template))
    return templates


def decide_match(
    scores: List[Tuple[str, float]],
    threshold: float,
    ambiguity_margin: float = 0.1,
    ambiguity_penalty: float = 0.1,
) -> Tuple[str, float]:
    """Pick the best label from scores sorted best-first, or UNKNOWN.

    If the two best people are within `ambiguity_margin` of each other the
    match is ambiguous, so the threshold is raised by `ambiguity_penalty`.
    """
    if not scores:
        return UNKNOWN, 0.0

    best_label, best_score = scores[0]
    effective_threshold = threshold
    if len(scores) >= 2 and (best_score - scores[1][1]) < ambiguity_margin:
        effective_threshold = threshold + ambiguity_penalty

    if best_score >= effective_threshold:
        return best_label, best_score
    return UNKNOWN, best_score


class FaceRecognizer:
    def __init__(
        self,
        embedder,
        database,
        threshold=0.6,
        min_quality=0.5,
        ambiguity_margin=0.1,
        ambiguity_penalty=0.1,
    ):
        self.embedder = embedder
        self.database = database
        self.threshold = threshold
        self.min_quality = min_quality
        self.ambiguity_margin = ambiguity_margin
        self.ambiguity_penalty = ambiguity_penalty

    def _get_mean_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Get mean embeddings for each person with validation."""
        return mean_embeddings_by_name(self.database.fetch_all_embeddings())

    def match(
        self, embedding: np.ndarray
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Match an embedding against the enrolled people.

        Returns (label, best_score, top_3_scores); label is "Unknown" when no
        one clears the threshold.
        """
        query = l2_normalize(embedding)
        if query is None:
            logger.warning("Query embedding has zero or invalid norm")
            return UNKNOWN, 0.0, []

        mean_embeddings = self._get_mean_embeddings()
        if not mean_embeddings:
            logger.warning("No valid embeddings in database")
            return UNKNOWN, 0.0, []

        # Templates and query are unit vectors, so cosine similarity is a dot product
        scores = [
            (label, float(np.dot(query, template)))
            for label, template in mean_embeddings
        ]
        scores.sort(key=lambda x: x[1], reverse=True)

        logger.debug("Top similarity scores: %s", scores[:3])

        label, best_score = decide_match(
            scores, self.threshold, self.ambiguity_margin, self.ambiguity_penalty
        )
        return label, best_score, scores[:3]

    def recognize(self, face: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Recognize a face with quality check and smart matching."""
        embedding, quality = self.embedder.get_embedding(face)

        if embedding is None or quality < self.min_quality:
            logger.warning("Face quality too low: %.2f", quality)
            return UNKNOWN, 0.0, []

        return self.match(embedding)

    def add_new_person(self, image: np.ndarray, name: str) -> bool:
        """Add a new person with quality checks."""
        embedding, quality = self.embedder.get_embedding(image)

        if embedding is None or quality < self.min_quality:
            logger.error("Cannot add person - face quality too low: %.2f", quality)
            return False

        added = self.database.insert_embedding(name, embedding.tolist(), None)
        if added:
            logger.info("Added new person '%s' to database", name)
        return added
