import base64
import os

import pytest

from api.utils import (
    ImageTooLargeError,
    InvalidImageError,
    decode_base64_image,
    normalize_person_name,
)
from app.settings import Settings
from tests.conftest import solid_image, to_b64


class TestSettings:
    def test_defaults_without_environment(self, monkeypatch):
        for key in list(os.environ):
            if key.startswith("FRCS_"):
                monkeypatch.delenv(key)
        assert Settings.from_env() == Settings()

    def test_reads_environment(self, monkeypatch):
        monkeypatch.setenv("FRCS_DB_PATH", "/data/faces.db")
        monkeypatch.setenv("FRCS_MATCH_THRESHOLD", "0.45")
        monkeypatch.setenv("FRCS_MAX_IMAGE_BYTES", "2048")
        monkeypatch.setenv("FRCS_LOG_RECOGNITIONS", "false")
        monkeypatch.setenv("FRCS_CORS_ORIGINS", "https://a.example, https://b.example,")
        monkeypatch.setenv("FRCS_API_KEY", "key")

        s = Settings.from_env()

        assert s.db_path == "/data/faces.db"
        assert s.match_threshold == 0.45
        assert s.max_image_bytes == 2048
        assert s.log_recognitions is False
        assert s.cors_origins == ("https://a.example", "https://b.example")
        assert s.api_key == "key"

    def test_empty_values_fall_back_to_defaults(self, monkeypatch):
        monkeypatch.setenv("FRCS_MATCH_THRESHOLD", "")
        assert Settings.from_env().match_threshold == Settings().match_threshold

    def test_invalid_numbers_name_the_variable(self, monkeypatch):
        monkeypatch.setenv("FRCS_MATCH_THRESHOLD", "high")
        with pytest.raises(ValueError, match="FRCS_MATCH_THRESHOLD"):
            Settings.from_env()


class TestDecodeBase64Image:
    def test_decodes_png_and_data_urls(self):
        image = solid_image((1, 2, 3), size=(8, 12))
        assert decode_base64_image(to_b64(image)).shape == (8, 12, 3)
        data_url = "data:image/png;base64," + to_b64(image)
        assert decode_base64_image(data_url).shape == (8, 12, 3)

    @pytest.mark.parametrize("value", [None, 5, "", "   ", "@@@@", "QUJD"])
    def test_rejects_non_images(self, value):
        with pytest.raises(InvalidImageError):
            decode_base64_image(value)

    def test_size_limit(self):
        payload = to_b64(solid_image((1, 2, 3), size=(64, 64)))
        raw_size = len(base64.b64decode(payload))
        assert decode_base64_image(payload, max_bytes=raw_size) is not None
        with pytest.raises(ImageTooLargeError):
            decode_base64_image(payload, max_bytes=raw_size - 1)


class TestNormalizePersonName:
    def test_collapses_whitespace(self):
        assert normalize_person_name("  Aya \t Nabil ") == "Aya Nabil"

    def test_allows_common_name_punctuation(self):
        assert normalize_person_name("O'Brien-Smith Jr.") == "O'Brien-Smith Jr."

    @pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "x" * 101, "名前<>", 3])
    def test_rejects_bad_names(self, bad):
        with pytest.raises(ValueError):
            normalize_person_name(bad)
