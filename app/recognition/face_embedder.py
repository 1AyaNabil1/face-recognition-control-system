import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from enum import Enum
import structlog
from pathlib import Path
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Initialize logger
logger = structlog.get_logger(__name__)


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    FACENET = "facenet"
    VGGFACE2 = "vggface2"
    INSIGHTFACE = "insightface"


@dataclass
class EmbeddingResult:
    """Result of face embedding generation."""

    embedding: np.ndarray
    model: str
    confidence: float
    inference_time: float


class FaceEmbedder:
    """
    Production-grade face embedding generator.
    Features:
    - Multiple model support
    - Normalized embeddings
    - Batch processing
    - Async-safe operations
    - Performance monitoring
    """

    def __init__(
        self,
        model_name: str = "facenet",
        device: Optional[str] = None,
        batch_size: int = 4,
        normalize: bool = True,
        fallback_model: Optional[str] = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_name = EmbeddingModel(model_name)
        self.batch_size = batch_size
        self.normalize = normalize
        self.fallback_model = fallback_model
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Performance tracking
        self.total_inferences = 0
        self.total_time = 0.0
        self.errors = 0

        # Initialize model
        self._init_model()

        logger.info(
            "Face embedder initialized",
            model=model_name,
            device=str(self.device),
            batch_size=batch_size,
        )

    def _init_model(self):
        """Initialize the embedding model."""
        try:
            if self.model_name in [EmbeddingModel.FACENET, EmbeddingModel.VGGFACE2]:
                from facenet_pytorch import InceptionResnetV1

                self.model = (
                    InceptionResnetV1(pretrained=self.model_name.value)
                    .eval()
                    .to(self.device)
                )
                self.embedding_size = 512
            elif self.model_name == EmbeddingModel.INSIGHTFACE:
                from insightface.app import FaceAnalysis

                self.model = FaceAnalysis(name="buffalo_l", root="./saved_models")
                self.model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
                self.embedding_size = 512
            else:
                raise ValueError(f"Unsupported model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error initializing model {self.model_name}", error=str(e))
            if self.fallback_model:
                logger.info(f"Falling back to {self.fallback_model}")
                self.model_name = EmbeddingModel(self.fallback_model)
                self._init_model()
            else:
                raise

    @torch.no_grad()
    def _generate_embedding(self, face: np.ndarray) -> np.ndarray:
        """Generate embedding for a single face."""
        try:
            # Convert to tensor
            if self.model_name in [EmbeddingModel.FACENET, EmbeddingModel.VGGFACE2]:
                face_tensor = torch.from_numpy(face).float()
                face_tensor = face_tensor.permute(2, 0, 1).unsqueeze(0)
                face_tensor = face_tensor.to(self.device)

                # Generate embedding
                embedding = self.model(face_tensor)

                # Normalize if requested
                if self.normalize:
                    embedding = F.normalize(embedding, p=2, dim=1)

                return embedding[0].cpu().numpy()

            elif self.model_name == EmbeddingModel.INSIGHTFACE:
                # InsightFace expects BGR format
                face_bgr = face[..., ::-1]
                result = self.model.get(face_bgr)[0]
                embedding = result.embedding

                # Normalize if requested
                if self.normalize:
                    embedding = embedding / np.linalg.norm(embedding)

                return embedding

        except Exception as e:
            logger.error("Embedding generation error", error=str(e))
            return np.zeros(self.embedding_size)

    async def generate_embedding(self, face: np.ndarray) -> EmbeddingResult:
        """
        Generate embedding for a single face asynchronously.
        Args:
            face: RGB face image array of shape (H, W, 3)
        Returns:
            EmbeddingResult with embedding and metadata
        """
        try:
            start_time = time.time()

            # Run embedding generation in thread pool
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._generate_embedding, face
            )

            inference_time = time.time() - start_time

            # Update metrics
            self.total_inferences += 1
            self.total_time += inference_time

            # Calculate confidence (placeholder - could be enhanced)
            confidence = float(np.mean(embedding != 0))

            return EmbeddingResult(
                embedding=embedding,
                model=self.model_name.value,
                confidence=confidence,
                inference_time=inference_time,
            )

        except Exception as e:
            self.errors += 1
            logger.error("Embedding generation error", error=str(e))
            raise

    async def generate_embeddings(
        self, faces: List[np.ndarray]
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for multiple faces in batches.
        Args:
            faces: List of RGB face image arrays
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        for i in range(0, len(faces), self.batch_size):
            batch = faces[i : i + self.batch_size]

            # Process batch in parallel
            tasks = [self.generate_embedding(face) for face in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    def compute_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray, metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings.
        Args:
            embedding1, embedding2: Face embeddings
            metric: Similarity metric ('cosine' or 'l2')
        Returns:
            Similarity score between 0 and 1
        """
        if metric == "cosine":
            similarity = np.dot(embedding1, embedding2)
            if self.normalize:
                return float(similarity)
            else:
                # Normalize on the fly
                return float(
                    similarity
                    / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
                )

        elif metric == "l2":
            distance = np.linalg.norm(embedding1 - embedding2)
            # Convert distance to similarity score
            return float(1 / (1 + distance))

        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def find_matches(
        self,
        embedding: np.ndarray,
        known_embeddings: List[np.ndarray],
        threshold: float = 0.6,
        metric: str = "cosine",
    ) -> List[Tuple[int, float]]:
        """
        Find matches for an embedding in a database of known embeddings.
        Args:
            embedding: Query embedding
            known_embeddings: List of known embeddings to match against
            threshold: Minimum similarity threshold
            metric: Similarity metric to use
        Returns:
            List of (index, similarity) tuples for matches above threshold
        """
        if not known_embeddings:
            return []

        # Convert to numpy array for batch computation
        known_embeddings = np.array(known_embeddings)

        if metric == "cosine":
            # Compute cosine similarities
            if self.normalize:
                similarities = np.dot(known_embeddings, embedding)
            else:
                similarities = np.dot(known_embeddings, embedding) / (
                    np.linalg.norm(known_embeddings, axis=1) * np.linalg.norm(embedding)
                )
        else:  # L2 distance
            distances = np.linalg.norm(known_embeddings - embedding, axis=1)
            similarities = 1 / (1 + distances)

        # Find matches above threshold
        matches = []
        for idx, similarity in enumerate(similarities):
            if similarity >= threshold:
                matches.append((idx, float(similarity)))

        # Sort by similarity
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def get_status(self) -> Dict:
        """Get embedder status and metrics."""
        return {
            "model": self.model_name.value,
            "device": str(self.device),
            "batch_size": self.batch_size,
            "normalize": self.normalize,
            "metrics": {
                "total_inferences": self.total_inferences,
                "average_time": self.total_time / max(1, self.total_inferences),
                "errors": self.errors,
            },
            "embedding_size": self.embedding_size,
        }
