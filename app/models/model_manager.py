import os
import logging
import requests
from pathlib import Path
from typing import Dict, Optional
import torch
from tqdm import tqdm
import hashlib
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Production model manager for downloading and caching AI models
    """

    MODEL_URLS = {
        "yolov8-face": "https://github.com/derronqi/yolov8-face/releases/download/v1.0/yolov8n-face-lindevs.pt",
        "antispoofing": "https://example.com/antispoofing.onnx",  # Replace with actual URL
        "facenet": "https://example.com/facenet.pt",  # Replace with actual URL
        "age-gender": "https://example.com/age-gender.h5",  # Replace with actual URL
        "emotion": "https://example.com/emotion.h5",  # Replace with actual URL
    }

    def __init__(self, model_dir: str = "/root/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.model_dir / "manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict:
        """Load model manifest with version info."""
        if self.manifest_path.exists():
            with open(self.manifest_path, "r") as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        """Save model manifest."""
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of model file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def download_model(self, model_name: str, force: bool = False) -> Optional[Path]:
        """
        Download and verify a model.

        Args:
            model_name: Name of model to download
            force: Force redownload even if exists

        Returns:
            Path to downloaded model or None if failed
        """
        if model_name not in self.MODEL_URLS:
            logger.error(f"Unknown model: {model_name}")
            return None

        url = self.MODEL_URLS[model_name]
        save_path = self.model_dir / f"{model_name}.pt"

        # Check if model exists and is valid
        if not force and save_path.exists():
            if model_name in self.manifest:
                current_hash = self._calculate_hash(save_path)
                if current_hash == self.manifest[model_name]["hash"]:
                    logger.info(f"Model {model_name} already exists and is valid")
                    return save_path

        # Download model
        try:
            logger.info(f"Downloading {model_name} from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024

            with (
                open(save_path, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    desc=f"Downloading {model_name}",
                ) as pbar,
            ):
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))

            # Verify and update manifest
            model_hash = self._calculate_hash(save_path)
            self.manifest[model_name] = {
                "hash": model_hash,
                "url": url,
                "downloaded_at": datetime.now().isoformat(),
                "size_bytes": os.path.getsize(save_path),
            }
            self._save_manifest()

            logger.info(f"Successfully downloaded {model_name}")
            return save_path

        except Exception as e:
            logger.error(f"Error downloading {model_name}: {str(e)}")
            if save_path.exists():
                save_path.unlink()
            return None

    def download_all(self, force: bool = False):
        """Download all required models."""
        success = True
        for model_name in self.MODEL_URLS:
            if not self.download_model(model_name, force):
                success = False

        if success:
            logger.info("All models downloaded successfully")
        else:
            logger.error("Some models failed to download")

    def get_model_info(self) -> Dict:
        """Get information about downloaded models."""
        info = {}
        for model_name in self.MODEL_URLS:
            model_path = self.model_dir / f"{model_name}.pt"
            if model_path.exists() and model_name in self.manifest:
                info[model_name] = {
                    "status": "downloaded",
                    "size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
                    "downloaded_at": self.manifest[model_name]["downloaded_at"],
                    "hash": self.manifest[model_name]["hash"][:8],  # First 8 chars
                }
            else:
                info[model_name] = {"status": "not_downloaded"}

        return info

    def verify_models(self) -> bool:
        """Verify all downloaded models are valid."""
        for model_name in self.MODEL_URLS:
            model_path = self.model_dir / f"{model_name}.pt"
            if not model_path.exists():
                logger.error(f"Model {model_name} is missing")
                return False

            if model_name not in self.manifest:
                logger.error(f"Model {model_name} has no manifest entry")
                return False

            current_hash = self._calculate_hash(model_path)
            if current_hash != self.manifest[model_name]["hash"]:
                logger.error(f"Model {model_name} hash mismatch")
                return False

        return True
