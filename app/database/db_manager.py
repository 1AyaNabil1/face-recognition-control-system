import sqlite3
import json
import csv
import os
from datetime import datetime


class EmbeddingDatabase:
    def __init__(self, db_path="app/database/embeddings.db", log_path="logs/recognition_log.csv"):
        self.db_path = db_path
        self.log_path = log_path
        self._ensure_directories()
        self.initialize()

    def _ensure_directories(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def initialize(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    image_path TEXT
                )
            """)
            conn.commit()

    def insert_embedding(self, name, embedding, image_path=None):
        emb_str = json.dumps(embedding)
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO embeddings (name, embedding, image_path) VALUES (?, ?, ?)",
                (name, emb_str, image_path),
            )
            conn.commit()

    def fetch_all_embeddings(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("SELECT name, embedding FROM embeddings")
            rows = c.fetchall()
        return [(name, json.loads(embedding)) for name, embedding in rows]

    def get_known_embeddings_and_labels(self):
        records = self.fetch_all_embeddings()
        if not records:
            return [], []
        labels = [name for name, _ in records]
        embeddings = [emb for _, emb in records]
        return embeddings, labels

    def log_recognition_event(self, name):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "N/A", name, "N/A"
            ])