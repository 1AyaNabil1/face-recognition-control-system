import numpy as np
import pytest

from app.recognition.face_recognizer import (
    UNKNOWN,
    FaceRecognizer,
    decide_match,
    l2_normalize,
    mean_embeddings_by_name,
)
from tests.conftest import DIM, unit_vector


class InMemoryDB:
    def __init__(self, records=(), insert_ok=True):
        self.records = list(records)
        self.insert_ok = insert_ok

    def fetch_all_embeddings(self):
        return list(self.records)

    def insert_embedding(self, name, embedding, image_path=None):
        if self.insert_ok:
            self.records.append((name, embedding))
        return self.insert_ok


class StubEmbedder:
    def __init__(self, embedding, quality=0.9):
        self.embedding = embedding
        self.quality = quality

    def get_embedding(self, face):
        return self.embedding, self.quality


def blend(a, b, weight):
    """Unit vector `weight` of the way from a towards b."""
    return l2_normalize((1 - weight) * a + weight * b)


class TestL2Normalize:
    def test_returns_unit_vector(self):
        v = l2_normalize(np.array([3.0, 4.0]))
        assert np.allclose(v, [0.6, 0.8])

    @pytest.mark.parametrize("bad", [np.zeros(4), np.array([np.nan, 1.0])])
    def test_rejects_vectors_without_direction(self, bad):
        assert l2_normalize(bad) is None


class TestMeanEmbeddings:
    def test_one_unit_template_per_person(self):
        a, b = unit_vector(1), unit_vector(2)
        templates = dict(
            mean_embeddings_by_name(
                [("alice", a.tolist()), ("alice", a.tolist()), ("bob", b.tolist())]
            )
        )
        assert set(templates) == {"alice", "bob"}
        assert np.allclose(templates["alice"], a)
        assert np.isclose(np.linalg.norm(templates["bob"]), 1.0)

    def test_samples_are_normalised_before_averaging(self):
        # A sample with a huge norm must not dominate the person's template
        a, b = unit_vector(1), unit_vector(2)
        templates = dict(
            mean_embeddings_by_name(
                [("alice", (100 * a).tolist()), ("alice", b.tolist())]
            )
        )
        template = templates["alice"]
        assert np.isclose(np.dot(template, a), np.dot(template, b))

    def test_skips_malformed_rows(self):
        good = unit_vector(3)
        templates = mean_embeddings_by_name(
            [("alice", [0.1] * 128), ("bob", [0.0] * DIM), ("carol", good.tolist())]
        )
        assert [name for name, _ in templates] == ["carol"]


class TestDecideMatch:
    def test_accepts_score_at_threshold(self):
        assert decide_match([("alice", 0.6), ("bob", 0.1)], threshold=0.6) == (
            "alice",
            0.6,
        )

    def test_below_threshold_is_unknown_but_reports_score(self):
        assert decide_match([("alice", 0.55)], threshold=0.6) == (UNKNOWN, 0.55)

    def test_ambiguous_top_two_needs_a_higher_score(self):
        # 0.65 clears 0.6, but bob is within the 0.1 margin -> threshold 0.7
        assert (
            decide_match([("alice", 0.65), ("bob", 0.6)], threshold=0.6)[0] == UNKNOWN
        )
        assert (
            decide_match([("alice", 0.75), ("bob", 0.7)], threshold=0.6)[0] == "alice"
        )

    def test_clear_margin_uses_base_threshold(self):
        assert (
            decide_match([("alice", 0.65), ("bob", 0.3)], threshold=0.6)[0] == "alice"
        )

    def test_penalty_can_be_disabled(self):
        result = decide_match(
            [("alice", 0.65), ("bob", 0.6)], threshold=0.6, ambiguity_penalty=0.0
        )
        assert result[0] == "alice"

    def test_no_scores(self):
        assert decide_match([], threshold=0.6) == (UNKNOWN, 0.0)


class TestFaceRecognizer:
    def setup_method(self):
        self.alice, self.bob = unit_vector(10), unit_vector(20)
        self.db = InMemoryDB(
            [("alice", self.alice.tolist()), ("bob", self.bob.tolist())]
        )

    def test_matches_the_closest_person(self):
        recognizer = FaceRecognizer(StubEmbedder(None), self.db, threshold=0.6)
        label, score, top = recognizer.match(blend(self.alice, self.bob, 0.2))
        assert label == "alice"
        assert score > 0.6
        assert [name for name, _ in top] == ["alice", "bob"]
        assert all(isinstance(s, float) for _, s in top)

    def test_unrelated_face_is_unknown(self):
        recognizer = FaceRecognizer(StubEmbedder(None), self.db, threshold=0.6)
        label, score, _ = recognizer.match(unit_vector(99))
        assert label == UNKNOWN
        assert score < 0.6

    def test_query_is_normalised(self):
        recognizer = FaceRecognizer(StubEmbedder(None), self.db, threshold=0.6)
        assert recognizer.match(50 * self.alice)[0] == "alice"

    def test_zero_query_and_empty_database_are_unknown(self):
        recognizer = FaceRecognizer(StubEmbedder(None), self.db)
        assert recognizer.match(np.zeros(DIM)) == (UNKNOWN, 0.0, [])
        empty = FaceRecognizer(StubEmbedder(None), InMemoryDB())
        assert empty.match(self.alice) == (UNKNOWN, 0.0, [])

    def test_recognize_rejects_low_quality_faces(self):
        recognizer = FaceRecognizer(
            StubEmbedder(self.alice, quality=0.2), self.db, min_quality=0.5
        )
        assert recognizer.recognize(np.zeros((10, 10, 3))) == (UNKNOWN, 0.0, [])

    def test_recognize_handles_missing_embedding(self):
        recognizer = FaceRecognizer(StubEmbedder(None, quality=0.0), self.db)
        assert recognizer.recognize(np.zeros((10, 10, 3)))[0] == UNKNOWN

    def test_add_new_person_reports_storage_failure(self):
        embedder = StubEmbedder(unit_vector(5))
        assert FaceRecognizer(embedder, InMemoryDB()).add_new_person(None, "carol")
        failing = FaceRecognizer(embedder, InMemoryDB(insert_ok=False))
        assert failing.add_new_person(None, "carol") is False
