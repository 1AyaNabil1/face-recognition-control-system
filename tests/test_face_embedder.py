from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app.embedding.face_embedder import CROP_PADDING, FaceEmbedder


def textured_face(size=112, seed=0):
    """Mid-grey image with sharp random texture: passes the quality gates."""
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, size=(size, size, 3), dtype=np.uint8)


def detected(bbox, det_score, embedding):
    return SimpleNamespace(
        bbox=np.array(bbox, dtype=float),
        det_score=det_score,
        embedding=np.asarray(embedding, dtype=np.float32),
    )


class FakeAnalyzer:
    """Stands in for insightface.app.FaceAnalysis."""

    def __init__(self, faces=(), error=None):
        self.faces = list(faces)
        self.error = error
        self.inputs = []

    def get(self, image):
        self.inputs.append(image)
        if self.error:
            raise self.error
        return self.faces


class TestCheckQuality:
    def test_accepts_a_reasonable_crop(self):
        assert FaceEmbedder.check_quality(textured_face()) is None

    @pytest.mark.parametrize(
        "image, reason",
        [
            (np.zeros((0, 0, 3), np.uint8), "empty face crop"),
            (np.full((112, 112), 128, np.uint8), "3-channel"),
            (np.full((112, 112, 3), 10, np.uint8), "too dark"),
            (np.full((112, 112, 3), 252, np.uint8), "too bright"),
            (np.full((112, 112, 3), 128, np.uint8), "contrast too low"),
        ],
    )
    def test_rejects_unusable_crops(self, image, reason):
        assert reason in FaceEmbedder.check_quality(image)

    def test_rejects_blurry_crops(self):
        # A smooth gradient has contrast but almost no high-frequency detail
        ramp = np.tile(np.linspace(40, 220, 112, dtype=np.uint8), (112, 1))
        blurry = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        assert FaceEmbedder.check_quality(blurry) == "image too blurry"

    def test_sharpness_does_not_depend_on_resolution(self):
        # Same content at 4x the size must not change the verdict
        small = textured_face(56)
        large = cv2.resize(small, (224, 224), interpolation=cv2.INTER_NEAREST)
        assert FaceEmbedder.check_quality(small) == FaceEmbedder.check_quality(large)


class TestGetEmbedding:
    def test_returns_unit_embedding_and_detection_score(self):
        analyzer = FakeAnalyzer([detected([10, 10, 90, 90], 0.87, [3.0] + [0.0] * 511)])
        embedder = FaceEmbedder(analyzer=analyzer)

        embedding, quality = embedder.get_embedding(textured_face())

        assert quality == pytest.approx(0.87)
        assert embedding.shape == (512,)
        assert np.linalg.norm(embedding) == pytest.approx(1.0)

    def test_pads_the_crop_so_insightface_can_find_the_face(self):
        analyzer = FakeAnalyzer([detected([0, 0, 50, 50], 0.9, np.ones(512))])
        FaceEmbedder(analyzer=analyzer).get_embedding(textured_face(100))
        pad = int(100 * CROP_PADDING)
        assert analyzer.inputs[0].shape == (100 + 2 * pad, 100 + 2 * pad, 3)

    def test_uses_the_largest_face_in_the_crop(self):
        small = detected([0, 0, 10, 10], 0.99, [1.0, 0.0] + [0.0] * 510)
        large = detected([0, 0, 80, 80], 0.80, [0.0, 1.0] + [0.0] * 510)
        embedder = FaceEmbedder(analyzer=FakeAnalyzer([small, large]))

        embedding, quality = embedder.get_embedding(textured_face())

        assert quality == pytest.approx(0.80)
        assert embedding[1] == pytest.approx(1.0)

    def test_no_face_found(self):
        embedder = FaceEmbedder(analyzer=FakeAnalyzer([]))
        assert embedder.get_embedding(textured_face()) == (None, 0.0)

    def test_low_detection_score_is_rejected_but_reported(self):
        analyzer = FakeAnalyzer([detected([0, 0, 80, 80], 0.3, np.ones(512))])
        embedder = FaceEmbedder(min_face_quality=0.5, analyzer=analyzer)
        embedding, quality = embedder.get_embedding(textured_face())
        assert embedding is None
        assert quality == pytest.approx(0.3)

    def test_quality_gate_runs_before_the_model(self):
        analyzer = FakeAnalyzer([detected([0, 0, 80, 80], 0.9, np.ones(512))])
        embedder = FaceEmbedder(analyzer=analyzer)
        dark = np.full((112, 112, 3), 5, np.uint8)
        assert embedder.get_embedding(dark) == (None, 0.0)
        assert analyzer.inputs == []

    def test_model_errors_do_not_propagate(self):
        embedder = FaceEmbedder(analyzer=FakeAnalyzer(error=RuntimeError("onnx")))
        assert embedder.get_embedding(textured_face()) == (None, 0.0)

    def test_zero_embedding_is_rejected(self):
        analyzer = FakeAnalyzer([detected([0, 0, 80, 80], 0.9, np.zeros(512))])
        embedding, _ = FaceEmbedder(analyzer=analyzer).get_embedding(textured_face())
        assert embedding is None
