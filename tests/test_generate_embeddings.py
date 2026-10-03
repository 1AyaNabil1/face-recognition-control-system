import importlib.util
from pathlib import Path

import cv2
import pytest

from api.recognition_service import RecognitionService
from app.database.db_manager import EmbeddingDatabase
from tests.conftest import FakeDetector, FakeEmbedder, solid_image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_embeddings.py"


def load_script():
    spec = importlib.util.spec_from_file_location("generate_embeddings", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def use_service(monkeypatch, module, factory):
    """Replace model loading in the script with `factory(settings)`."""
    monkeypatch.setattr(
        module.RecognitionService,
        "from_settings",
        classmethod(lambda cls, s=None: factory(s)),
    )


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    for folder in ("Alice", "Bob", "<bad>"):
        (root / folder).mkdir(parents=True)
    cv2.imwrite(str(root / "Alice" / "1.png"), solid_image((0, 0, 255)))
    cv2.imwrite(str(root / "Alice" / "2.png"), solid_image((0, 0, 250)))
    cv2.imwrite(str(root / "Bob" / "1.png"), solid_image((255, 0, 0)))
    cv2.imwrite(str(root / "<bad>" / "1.png"), solid_image((0, 255, 0)))
    (root / "Alice" / "notes.txt").write_text("not an image")
    (root / "Bob" / "broken.jpg").write_bytes(b"not really a jpeg")
    (root / "README.md").write_text("not a person folder")
    return str(root)


def test_enrolls_every_person_folder(dataset, monkeypatch, make_service):
    module = load_script()
    service = make_service()
    use_service(monkeypatch, module, lambda s: service)

    added, skipped = module.generate_embeddings(dataset)

    # broken.jpg is skipped, notes.txt ignored, "<bad>" is not a valid name
    assert (added, skipped) == (3, 1)
    assert service.db.list_people() == [("Alice", 2), ("Bob", 1)]


def test_rejected_faces_are_skipped(dataset, monkeypatch, make_service):
    module = load_script()
    service = make_service(
        detector=FakeDetector(boxes=[]), embedder=FakeEmbedder(reject_reason="blurry")
    )
    use_service(monkeypatch, module, lambda s: service)

    assert module.generate_embeddings(dataset) == (0, 4)
    assert service.db.list_people() == []


def test_existing_database_is_kept_unless_reset(dataset, monkeypatch, settings):
    module = load_script()
    monkeypatch.setenv("FRCS_DB_PATH", settings.db_path)
    monkeypatch.setenv("FRCS_RECOGNITION_LOG_PATH", settings.recognition_log_path)

    def open_db():
        return EmbeddingDatabase(settings.db_path, settings.recognition_log_path)

    open_db().insert_embedding("Carol", [0.1] * 512)

    def real_db_service(s):
        return RecognitionService(
            detector=FakeDetector(),
            embedder=FakeEmbedder(),
            database=EmbeddingDatabase(s.db_path, s.recognition_log_path),
            settings=s,
        )

    use_service(monkeypatch, module, real_db_service)

    module.generate_embeddings(dataset)
    people = dict(open_db().list_people())
    assert people == {"Alice": 2, "Bob": 1, "Carol": 1}

    module.generate_embeddings(dataset, reset=True)
    people = dict(open_db().list_people())
    assert people == {"Alice": 2, "Bob": 1}
