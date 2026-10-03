import base64
import csv
import os
from dataclasses import replace

import pytest

from api.app import create_app
from tests.conftest import FakeDetector, FakeEmbedder, solid_image, to_b64

RED, BLUE = (0, 0, 255), (255, 0, 0)


def enroll(client, name, colour, **headers):
    return client.post(
        "/api/add-person",
        json={"name": name, "image": to_b64(solid_image(colour))},
        headers=headers,
    )


def recognize(client, colour, **extra):
    return client.post(
        "/api/recognize", json={"image": to_b64(solid_image(colour)), **extra}
    )


class TestHealth:
    def test_reports_ok_and_model_state(self, client, settings):
        body = client.get("/api/health").get_json()
        assert body["status"] == "OK"
        assert body["models_loaded"] is True

        # Without an injected service nothing is loaded until the first request
        lazy = create_app(settings=settings).test_client()
        assert lazy.get("/api/health").get_json()["models_loaded"] is False


class TestRecognize:
    def test_enroll_then_recognize(self, client):
        assert enroll(client, "Alice", RED).status_code == 200
        assert enroll(client, "Bob", BLUE).status_code == 200

        response = recognize(client, RED)
        assert response.status_code == 200
        body = response.get_json()
        assert body["result"] == "Alice"
        assert body["confidence"] == pytest.approx(1.0, abs=1e-3)
        assert [m["name"] for m in body["top_matches"]] == ["Alice", "Bob"]
        assert body["faces_detected"] == 1
        assert body["face_box"] == [20, 20, 100, 100]
        assert base64.b64decode(body["annotated_image"])

    def test_unknown_face(self, client):
        enroll(client, "Alice", RED)
        body = recognize(client, (0, 255, 0)).get_json()
        assert body["result"] == "Unknown"

    def test_annotated_image_is_optional(self, client):
        body = recognize(client, RED, return_annotated_image=False).get_json()
        assert "annotated_image" not in body

    def test_no_face_detected(self, make_client):
        client = make_client(detector=FakeDetector(boxes=[]))
        body = recognize(client, RED).get_json()
        assert body["result"] == "Unknown"
        assert body["faces_detected"] == 0
        assert body["face_box"] is None

    def test_recognitions_are_logged_only_for_known_people(self, client, settings):
        enroll(client, "Alice", RED)
        recognize(client, RED)
        recognize(client, (0, 255, 0))
        with open(settings.recognition_log_path, newline="") as f:
            assert [row[1] for row in csv.reader(f)] == ["Alice"]

    def test_logging_can_be_disabled(self, make_client, settings):
        quiet = replace(settings, log_recognitions=False)
        client = make_client(app_settings=quiet)
        enroll(client, "Alice", RED)
        recognize(client, RED)
        assert not os.path.exists(settings.recognition_log_path)


class TestRequestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"data": "not json", "content_type": "text/plain"},
            {"json": ["a list"]},
            {"json": {"img": "wrong key"}},
        ],
    )
    def test_missing_or_malformed_body(self, client, kwargs):
        response = client.post("/api/recognize", **kwargs)
        assert response.status_code == 400
        assert response.get_json() == {"error": "No image provided"}

    @pytest.mark.parametrize(
        "image, message",
        [
            (123, "non-empty base64 string"),
            ("", "non-empty base64 string"),
            ("not base64 at all!", "not valid base64"),
            (base64.b64encode(b"hello world").decode(), "could not be decoded"),
        ],
    )
    def test_bad_images_are_rejected(self, client, image, message):
        response = client.post("/api/recognize", json={"image": image})
        assert response.status_code == 400
        assert message in response.get_json()["error"]

    def test_accepts_data_urls(self, client):
        data_url = "data:image/png;base64," + to_b64(solid_image(RED))
        response = client.post("/api/recognize", json={"image": data_url})
        assert response.status_code == 200

    def test_oversized_image(self, make_client, settings):
        client = make_client(app_settings=replace(settings, max_image_bytes=1000))
        noisy = os.urandom(3000)
        response = client.post(
            "/api/recognize", json={"image": base64.b64encode(noisy).decode()}
        )
        assert response.status_code == 413
        assert "larger than" in response.get_json()["error"]

    def test_oversized_request_body(self, make_client, settings):
        client = make_client(app_settings=replace(settings, max_image_bytes=1000))
        response = client.post("/api/recognize", json={"image": "A" * 100_000})
        assert response.status_code == 413
        assert "error" in response.get_json()

    @pytest.mark.parametrize(
        "name",
        ["", "   ", "x" * 101, "../../etc/passwd", "<script>", "a/b", 42, None],
    )
    def test_invalid_names_are_rejected(self, client, name):
        response = client.post(
            "/api/add-person", json={"name": name, "image": to_b64(solid_image(RED))}
        )
        assert response.status_code == 400

    @pytest.mark.parametrize(
        "raw, stored", [("  José   Ünal ", "José Ünal"), ("آية نبيل", "آية نبيل")]
    )
    def test_unicode_names_are_accepted_and_normalised(self, client, raw, stored):
        response = enroll(client, raw, RED)
        assert response.status_code == 200
        assert response.get_json()["name"] == stored

    def test_unknown_route_and_wrong_method_return_json(self, client):
        response = client.get("/nope")
        assert response.status_code == 404
        assert "error" in response.get_json()
        response = client.get("/api/recognize")
        assert response.status_code == 405
        assert "error" in response.get_json()


class TestEnrollment:
    def test_requires_name_and_image(self, client):
        response = client.post("/api/add-person", json={"name": "Alice"})
        assert response.status_code == 400

    def test_no_face_is_unprocessable(self, make_client):
        client = make_client(detector=FakeDetector(boxes=[]))
        response = enroll(client, "Alice", RED)
        assert response.status_code == 422
        assert "no face detected" in response.get_json()["error"]

    def test_quality_rejection_reason_is_returned(self, make_client):
        client = make_client(embedder=FakeEmbedder(reject_reason="image too blurry"))
        response = enroll(client, "Alice", RED)
        assert response.status_code == 422
        assert "image too blurry" in response.get_json()["error"]
        assert client.get("/api/persons").get_json() == {"persons": []}

    def test_uses_the_largest_detected_face(self, make_client):
        client = make_client(
            detector=FakeDetector(boxes=[(0, 0, 40, 40), (10, 10, 110, 110)])
        )
        body = enroll(client, "Alice", RED).get_json()
        assert body["face_box"] == [10, 10, 110, 110]


class TestPersons:
    def test_list_and_delete(self, client):
        enroll(client, "Alice", RED)
        enroll(client, "Alice", RED)
        enroll(client, "Bob", BLUE)
        assert client.get("/api/persons").get_json() == {
            "persons": [
                {"name": "Alice", "embeddings": 2},
                {"name": "Bob", "embeddings": 1},
            ]
        }

        response = client.delete("/api/persons/Alice")
        assert response.status_code == 200
        assert response.get_json()["deleted"] == 2
        # Deleted people can no longer be recognised
        assert recognize(client, RED).get_json()["result"] == "Unknown"

    def test_work_without_loading_models(self, settings, database):
        # Erasure must not depend on the face models being available
        database.insert_embedding("Alice", [0.1] * 512)
        client = create_app(settings=settings).test_client()
        assert client.get("/api/persons").get_json()["persons"][0]["name"] == "Alice"
        assert client.delete("/api/persons/Alice").status_code == 200
        assert client.get("/api/health").get_json()["models_loaded"] is False

    def test_delete_unknown_person(self, client):
        assert client.delete("/api/persons/Nobody").status_code == 404

    def test_delete_name_with_space(self, client):
        enroll(client, "Aya Nabil", RED)
        assert client.delete("/api/persons/Aya%20Nabil").status_code == 200


class TestApiKey:
    @pytest.fixture
    def secured(self, make_client, settings):
        return make_client(app_settings=replace(settings, api_key="s3cret"))

    def test_rejects_missing_or_wrong_key(self, secured):
        assert recognize(secured, RED).status_code == 401
        response = secured.get("/api/persons", headers={"X-API-Key": "wrong"})
        assert response.status_code == 401
        assert response.get_json() == {"error": "missing or invalid API key"}

    def test_accepts_the_right_key(self, secured):
        assert (
            enroll(secured, "Alice", RED, **{"X-API-Key": "s3cret"}).status_code == 200
        )
        response = secured.get("/api/persons", headers={"X-API-Key": "s3cret"})
        assert response.status_code == 200

    def test_health_stays_public(self, secured):
        assert secured.get("/api/health").status_code == 200


class TestErrorHandling:
    def test_internal_errors_do_not_leak_details(self, make_client):
        class ExplodingDetector(FakeDetector):
            def detect_faces(self, image):
                raise RuntimeError("database password is hunter2")

        client = make_client(detector=ExplodingDetector())
        response = recognize(client, RED)
        assert response.status_code == 500
        assert response.get_json() == {"error": "Internal server error"}

    def test_model_loading_failure_is_503(self, settings):
        broken = replace(settings, yolo_model_path="does/not/exist.pt")
        client = create_app(settings=broken).test_client()
        response = recognize(client, RED)
        assert response.status_code == 503
        assert "could not be loaded" in response.get_json()["error"]


class TestCors:
    def test_disabled_by_default(self, client):
        response = client.get("/api/health", headers={"Origin": "https://evil.example"})
        assert "Access-Control-Allow-Origin" not in response.headers

    def test_allows_configured_origins_only(self, make_client, settings):
        client = make_client(
            app_settings=replace(settings, cors_origins=("https://app.example",))
        )
        ok = client.get("/api/health", headers={"Origin": "https://app.example"})
        assert ok.headers["Access-Control-Allow-Origin"] == "https://app.example"
        other = client.get("/api/health", headers={"Origin": "https://evil.example"})
        assert "Access-Control-Allow-Origin" not in other.headers
