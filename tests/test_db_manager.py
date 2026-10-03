import csv
import sqlite3

import numpy as np
import pytest

from app.database.db_manager import EmbeddingDatabase
from tests.conftest import DIM, unit_vector


def test_insert_and_fetch_round_trip(database):
    emb = unit_vector(1).tolist()
    assert database.insert_embedding("alice", emb, "photos/alice.jpg")
    [(name, stored)] = database.fetch_all_embeddings()
    assert name == "alice"
    assert np.allclose(stored, emb)


@pytest.mark.parametrize(
    "name, embedding",
    [
        ("", [0.1] * DIM),
        (None, [0.1] * DIM),
        ("alice", [0.1] * 128),
        ("alice", [0.1] * (DIM - 1) + [float("nan")]),
        ("alice", [0.1] * (DIM - 1) + [float("inf")]),
        ("alice", ["x"] * DIM),
        ("alice", np.ones(DIM)),  # must be a plain list, as stored in JSON
    ],
)
def test_insert_rejects_invalid_input(database, name, embedding):
    assert database.insert_embedding(name, embedding) is False
    assert database.fetch_all_embeddings() == []


def test_list_and_delete_people(database):
    for seed in (1, 2):
        database.insert_embedding("alice", unit_vector(seed).tolist())
    database.insert_embedding("bob", unit_vector(3).tolist())

    assert database.list_people() == [("alice", 2), ("bob", 1)]
    assert database.delete_person("alice") == 2
    assert database.list_people() == [("bob", 1)]
    assert database.delete_person("alice") == 0


def test_fetch_skips_and_cleanup_removes_corrupt_rows(database):
    database.insert_embedding("alice", unit_vector(1).tolist())
    with sqlite3.connect(database.db_path) as conn:
        conn.execute(
            "INSERT INTO embeddings (name, embedding) VALUES (?, ?)", ("bad", "{oops")
        )
        conn.execute(
            "INSERT INTO embeddings (name, embedding) VALUES (?, ?)",
            ("short", "[1, 2]"),
        )

    assert [name for name, _ in database.fetch_all_embeddings()] == ["alice"]
    assert database.cleanup_invalid_entries() == 2
    assert database.list_people() == [("alice", 1)]


def test_recognition_log_and_usage_stats(database):
    database.insert_embedding("alice", unit_vector(1).tolist())
    database.log_recognition_event("alice")
    database.log_recognition_event("alice")

    with open(database.log_path, newline="") as f:
        rows = list(csv.reader(f))
    assert [row[1] for row in rows] == ["alice", "alice"]

    with sqlite3.connect(database.db_path) as conn:
        (use_count,) = conn.execute(
            "SELECT use_count FROM embeddings WHERE name = 'alice'"
        ).fetchone()
    assert use_count == 2


def test_paths_without_a_directory_work(tmp_path, monkeypatch):
    # os.makedirs("") used to raise for a bare file name
    monkeypatch.chdir(tmp_path)
    db = EmbeddingDatabase(db_path="embeddings.db", log_path="recognitions.csv")
    assert db.insert_embedding("alice", unit_vector(1).tolist())
    assert (tmp_path / "embeddings.db").exists()


def test_creates_missing_directories(tmp_path):
    db = EmbeddingDatabase(
        db_path=str(tmp_path / "a" / "b" / "embeddings.db"),
        log_path=str(tmp_path / "c" / "log.csv"),
    )
    db.log_recognition_event("nobody")
    assert (tmp_path / "a" / "b" / "embeddings.db").exists()
    assert (tmp_path / "c" / "log.csv").exists()
