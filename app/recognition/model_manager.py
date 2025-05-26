import os
import torch
import torch.nn as nn
from typing import Dict, Optional, List, Any, Union
import structlog
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from facenet_pytorch import MTCNN, InceptionResnetV1
from insightface.app import FaceAnalysis
from deepface import DeepFace
import onnxruntime as ort
import numpy as np

# Initialize logger
logger = structlog.get_logger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""

    load_time: float
    last_inference_time: Optional[float]
    memory_usage: Optional[int]
    total_inferences: int
    avg_inference_time: float
    errors: int


@dataclass
class ModelConfig:
    """Model configuration."""

    name: str
    type: str
    version: str
    enabled: bool
    device: str
    batch_size: int
    params: Dict[str, Any]


class ModelManager:
    """
    Production model manager with lazy loading and monitoring.
    Features:
    - Lazy model loading
    - GPU/CPU fallback
    - Performance monitoring
    - Resource management
    - Error tracking
    """

    def __init__(
        self,
        model_path: str = "models/",
        device: Optional[str] = None,
        batch_size: int = 4,
    ):
        # Initialize settings
        self.model_path = Path(model_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Model storage
        self.models: Dict[str, Any] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.metrics: Dict[str, ModelMetrics] = {}

        # Define model configurations
        self._init_model_configs()

        logger.info(
            "Model manager initialized",
            device=str(self.device),
            models=list(self.configs.keys()),
        )

    def _init_model_configs(self):
        """Initialize model configurations."""
        self.configs = {
            "mtcnn": ModelConfig(
                name="MTCNN",
                type="detection",
                version="2.0.0",
                enabled=True,
                device=str(self.device),
                batch_size=self.batch_size,
                params={
                    "image_size": 160,
                    "margin": 0,
                    "min_face_size": 20,
                    "thresholds": [0.6, 0.7, 0.7],
                    "factor": 0.709,
                    "post_process": False,
                    "keep_all": True,
                },
            ),
            "facenet": ModelConfig(
                name="FaceNet",
                type="recognition",
                version="vggface2",
                enabled=True,
                device=str(self.device),
                batch_size=self.batch_size,
                params={"pretrained": "vggface2", "classify": False},
            ),
            "antispoofing": ModelConfig(
                name="AntiSpoofing",
                type="liveness",
                version="1.0.0",
                enabled=True,
                device=str(self.device),
                batch_size=1,
                params={
                    "model_path": str(self.model_path / "antispoofing.onnx"),
                    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                },
            ),
            "attributes": ModelConfig(
                name="Attributes",
                type="analysis",
                version="2.0.0",
                enabled=True,
                device="cpu",  # DeepFace models run on CPU
                batch_size=1,
                params={},
            ),
        }

    async def load_models(self, models: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Load specified models or all enabled models.
        Returns dict of model names and their load status.
        """
        models_to_load = models or [
            name for name, config in self.configs.items() if config.enabled
        ]

        results = {}
        for name in models_to_load:
            if name not in self.configs:
                logger.warning(f"Unknown model: {name}")
                results[name] = False
                continue

            try:
                await self._load_model(name)
                results[name] = True
            except Exception as e:
                logger.error(f"Error loading model {name}", error=str(e))
                results[name] = False

        return results

    async def _load_model(self, name: str):
        """Load a specific model with performance tracking."""
        if name in self.models:
            logger.info(f"Model {name} already loaded")
            return

        config = self.configs[name]
        start_time = time.time()

        try:
            if name == "mtcnn":
                model = MTCNN(device=self.device, **config.params)
            elif name == "facenet":
                model = InceptionResnetV1(**config.params).to(self.device).eval()
            elif name == "antispoofing":
                model = ort.InferenceSession(**config.params)
            elif name == "attributes":
                model = {
                    "age": DeepFace.build_model("Age"),
                    "gender": DeepFace.build_model("Gender"),
                    "emotion": DeepFace.build_model("Emotion"),
                }
            else:
                raise ValueError(f"Unknown model type: {name}")

            self.models[name] = model
            load_time = time.time() - start_time

            # Initialize metrics
            self.metrics[name] = ModelMetrics(
                load_time=load_time,
                last_inference_time=None,
                memory_usage=self._get_model_memory(model)
                if hasattr(model, "parameters")
                else None,
                total_inferences=0,
                avg_inference_time=0.0,
                errors=0,
            )

            logger.info(
                f"Model {name} loaded successfully",
                load_time=load_time,
                device=config.device,
            )

        except Exception as e:
            logger.error(f"Error loading model {name}", error=str(e))
            raise

    def _get_model_memory(self, model: nn.Module) -> int:
        """Calculate model memory usage in bytes."""
        return sum(p.numel() * p.element_size() for p in model.parameters())

    async def unload_models(
        self, models: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Unload specified models or all loaded models.
        Returns dict of model names and their unload status.
        """
        models_to_unload = models or list(self.models.keys())
        results = {}

        for name in models_to_unload:
            try:
                if name in self.models:
                    model = self.models[name]

                    # Clean up based on model type
                    if hasattr(model, "to"):
                        model.to("cpu")
                    elif isinstance(model, dict):
                        model.clear()

                    del self.models[name]
                    results[name] = True

                    logger.info(f"Model {name} unloaded successfully")
                else:
                    results[name] = False
            except Exception as e:
                logger.error(f"Error unloading model {name}", error=str(e))
                results[name] = False

        # Clear CUDA cache if no models are loaded
        if not self.models and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    async def get_model(self, name: str, load_if_needed: bool = True) -> Optional[Any]:
        """
        Get a model by name, optionally loading it if not loaded.
        """
        if name not in self.configs:
            raise ValueError(f"Unknown model: {name}")

        if name not in self.models and load_if_needed:
            await self._load_model(name)

        return self.models.get(name)

    @torch.no_grad()
    async def run_inference(
        self, name: str, inputs: Union[torch.Tensor, np.ndarray], **kwargs
    ) -> Any:
        """
        Run inference on a model with performance tracking.
        """
        model = await self.get_model(name)
        if not model:
            raise ValueError(f"Model {name} not loaded")

        try:
            start_time = time.time()

            if isinstance(model, nn.Module):
                # PyTorch model
                if isinstance(inputs, np.ndarray):
                    inputs = torch.from_numpy(inputs).to(self.device)
                result = model(inputs, **kwargs)
            elif isinstance(model, ort.InferenceSession):
                # ONNX model
                result = model.run(None, {"input": inputs})[0]
            elif isinstance(model, dict):
                # DeepFace models
                result = {k: m.predict(inputs) for k, m in model.items()}
            else:
                raise ValueError(f"Unsupported model type for {name}")

            # Update metrics
            inference_time = time.time() - start_time
            metrics = self.metrics[name]
            metrics.last_inference_time = inference_time
            metrics.total_inferences += 1
            metrics.avg_inference_time = (
                metrics.avg_inference_time * (metrics.total_inferences - 1)
                + inference_time
            ) / metrics.total_inferences

            return result

        except Exception as e:
            self.metrics[name].errors += 1
            logger.error(f"Inference error for model {name}", error=str(e))
            raise

    def get_status(self) -> Dict:
        """Get detailed status of all models."""
        return {
            "status": "healthy" if self.models else "inactive",
            "device": str(self.device),
            "models": {
                name: {
                    "loaded": name in self.models,
                    "config": asdict(config),
                    "metrics": asdict(
                        self.metrics.get(
                            name,
                            ModelMetrics(
                                load_time=0.0,
                                last_inference_time=None,
                                memory_usage=None,
                                total_inferences=0,
                                avg_inference_time=0.0,
                                errors=0,
                            ),
                        )
                    ),
                }
                for name, config in self.configs.items()
            },
            "resources": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0,
                "cuda_memory": {
                    "allocated": torch.cuda.memory_allocated()
                    if torch.cuda.is_available()
                    else 0,
                    "cached": torch.cuda.memory_reserved()
                    if torch.cuda.is_available()
                    else 0,
                },
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
