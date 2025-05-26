import sqlite3
import json
import csv
import os
from datetime import datetime
import numpy as np
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EmbeddingDatabase:
    def __init__(
        self, db_path="app/database/embeddings.db", log_path="logs/recognition_log.csv"
    ):
        self.db_path = db_path
        self.log_path = log_path
        self._ensure_directories()
        self.initialize()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _validate_embedding(self, embedding: List[float]) -> bool:
        """Validate embedding format and values."""
        try:
            if not isinstance(embedding, list):
                return False

            if len(embedding) != 512:  # Expected embedding size
                return False

            # Check for valid float values
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False

            # Check for NaN or Inf values
            if any(np.isnan(x) or np.isinf(x) for x in embedding):
                return False

            return True
        except Exception as e:
            logger.error(f"Embedding validation error: {e}")
            return False

    def initialize(self):
        """Initialize database with proper schema."""
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()

            # Create embeddings table with constraints
            c.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    image_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    CONSTRAINT valid_name CHECK(length(name) > 0)
                )
            """)

            # Create index for faster lookups
            c.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_name 
                ON embeddings(name)
            """)

            conn.commit()

    def insert_embedding(
        self, name: str, embedding: List[float], image_path: Optional[str] = None
    ) -> bool:
        """Insert new embedding with validation."""
        if not name or not isinstance(name, str):
            logger.error("Invalid name provided")
            return False

        if not self._validate_embedding(embedding):
            logger.error("Invalid embedding format")
            return False

        try:
            emb_str = json.dumps(embedding)
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    INSERT INTO embeddings (name, embedding, image_path) 
                    VALUES (?, ?, ?)
                    """,
                    (name, emb_str, image_path),
                )
                conn.commit()
            logger.info(f"Successfully added embedding for {name}")
            return True
        except Exception as e:
            logger.error(f"Error inserting embedding: {e}")
            return False

    def fetch_all_embeddings(self) -> List[Tuple[str, List[float]]]:
        """Fetch all embeddings with validation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT name, embedding FROM embeddings")
                rows = c.fetchall()

            valid_records = []
            for name, emb_str in rows:
                try:
                    embedding = json.loads(emb_str)
                    if self._validate_embedding(embedding):
                        valid_records.append((name, embedding))
                    else:
                        logger.warning(f"Invalid embedding found for {name}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode embedding for {name}")

            return valid_records
        except Exception as e:
            logger.error(f"Error fetching embeddings: {e}")
            return []

    def update_usage_stats(self, name: str):
        """Update usage statistics for recognized faces."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute(
                    """
                    UPDATE embeddings 
                    SET last_used = CURRENT_TIMESTAMP,
                        use_count = use_count + 1
                    WHERE name = ?
                    """,
                    (name,),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating usage stats: {e}")

    def cleanup_invalid_entries(self) -> int:
        """Remove invalid entries from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("SELECT id, name, embedding FROM embeddings")
                rows = c.fetchall()

                deleted_count = 0
                for id_, name, emb_str in rows:
                    try:
                        embedding = json.loads(emb_str)
                        if not self._validate_embedding(embedding):
                            c.execute("DELETE FROM embeddings WHERE id = ?", (id_,))
                            deleted_count += 1
                            logger.info(f"Removed invalid embedding for {name}")
                    except json.JSONDecodeError:
                        c.execute("DELETE FROM embeddings WHERE id = ?", (id_,))
                        deleted_count += 1
                        logger.info(f"Removed corrupted embedding for {name}")

                conn.commit()
                return deleted_count
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0

    def log_recognition_event(self, name: str):
        """Log recognition events to CSV."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, name])

            self.update_usage_stats(name)
        except Exception as e:
            logger.error(f"Error logging recognition event: {e}")
