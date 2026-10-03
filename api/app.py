"""Flask API for the YOLOv8 + InsightFace recognition pipeline.

Run locally with `python -m api.app` or in production with
`gunicorn "api.app:create_app()"`. Configuration comes from environment
variables, see `.env.example`.
"""

import hmac
import logging
import os
import threading

from flask import Flask, current_app, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, ServiceUnavailable

from api.health import health_bp
from api.utils import (
    ImageTooLargeError,
    InvalidImageError,
    decode_base64_image,
    encode_image_base64,
    normalize_person_name,
)
from app.database.db_manager import EmbeddingDatabase
from app.settings import Settings, load_dotenv_if_available

logger = logging.getLogger(__name__)

# Routes that never require the API key
PUBLIC_ENDPOINTS = {"health.health_check"}


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def get_service():
    """Return the recognition service, loading the models on first use."""
    state = current_app.extensions["frcs"]
    if state["service"] is None:
        with state["lock"]:
            if state["service"] is None:
                from api.recognition_service import RecognitionService

                logger.info("Loading face detection and embedding models")
                try:
                    state["service"] = RecognitionService.from_settings(
                        state["settings"]
                    )
                except Exception as exc:
                    logger.exception("Could not load the recognition models")
                    raise ServiceUnavailable(
                        "Recognition models could not be loaded; check the server logs"
                    ) from exc
    return state["service"]


def get_database():
    """Return the embeddings database without loading any model.

    Listing and deleting people only touches SQLite, so erasure requests keep
    working even when the models are unavailable.
    """
    state = current_app.extensions["frcs"]
    if state["service"] is not None:
        return state["service"].db
    if state["database"] is None:
        with state["lock"]:
            if state["database"] is None:
                settings = state["settings"]
                state["database"] = EmbeddingDatabase(
                    db_path=settings.db_path, log_path=settings.recognition_log_path
                )
    return state["database"]


def _decode_image_field(data):
    settings = current_app.extensions["frcs"]["settings"]
    return decode_base64_image(data.get("image"), max_bytes=settings.max_image_bytes)


def _json_body():
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else None


def create_app(settings: Settings = None, service=None) -> Flask:
    """Create the Flask app.

    `service` can be injected (tests pass a fake); otherwise the real models
    are loaded lazily on the first request that needs them.
    """
    if settings is None:
        load_dotenv_if_available()
        settings = Settings.from_env()

    if not logging.getLogger().handlers:
        logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    app = Flask(__name__)
    # Base64 inflates the image by ~4/3; leave headroom for the JSON wrapper.
    app.config["MAX_CONTENT_LENGTH"] = settings.max_image_bytes * 4 // 3 + 64 * 1024
    app.extensions["frcs"] = {
        "settings": settings,
        "service": service,
        "database": None,
        "lock": threading.Lock(),
    }

    if settings.cors_origins:
        CORS(app, origins=list(settings.cors_origins))

    if not settings.api_key:
        logger.warning(
            "FRCS_API_KEY is not set: the API accepts unauthenticated requests. "
            "Set it before exposing the service beyond localhost."
        )

    app.register_blueprint(health_bp)

    @app.before_request
    def require_api_key():
        if not settings.api_key or request.method == "OPTIONS":
            return None
        if request.endpoint is None or request.endpoint in PUBLIC_ENDPOINTS:
            return None
        supplied = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(supplied.encode(), settings.api_key.encode()):
            return _error("missing or invalid API key", 401)
        return None

    @app.route("/api/recognize", methods=["POST"])
    def recognize_face():
        data = _json_body()
        if data is None or "image" not in data:
            return _error("No image provided", 400)

        image = _decode_image_field(data)
        result = get_service().recognize_image(image)

        response = {
            "result": result.name,
            "confidence": round(result.score, 4),
            "top_matches": [
                {"name": label, "score": round(s, 4)} for label, s in result.top_matches
            ],
            "faces_detected": result.faces_detected,
            "face_box": list(result.face_box) if result.face_box else None,
        }
        if data.get("return_annotated_image", True):
            response["annotated_image"] = encode_image_base64(result.annotated_image)
        return jsonify(response), 200

    @app.route("/api/add-person", methods=["POST"])
    def add_person():
        data = _json_body()
        if data is None or "name" not in data or "image" not in data:
            return _error("Name or image missing", 400)

        try:
            name = normalize_person_name(data["name"])
        except ValueError as exc:
            return _error(str(exc), 400)

        image = _decode_image_field(data)
        result = get_service().enroll(image, name)
        if not result.success:
            return _error(f"Could not add person: {result.reason}", 422)

        return jsonify(
            {
                "message": "Person added successfully!",
                "name": name,
                "quality": round(result.quality, 4),
                "face_box": list(result.face_box) if result.face_box else None,
            }
        ), 200

    @app.route("/api/persons", methods=["GET"])
    def list_persons():
        people = get_database().list_people()
        return jsonify(
            {"persons": [{"name": name, "embeddings": n} for name, n in people]}
        ), 200

    @app.route("/api/persons/<path:name>", methods=["DELETE"])
    def delete_person(name):
        try:
            name = normalize_person_name(name)
        except ValueError as exc:
            return _error(str(exc), 400)

        deleted = get_database().delete_person(name)
        if not deleted:
            return _error("Person not found", 404)
        return jsonify({"message": "Person deleted", "name": name, "deleted": deleted})

    @app.errorhandler(ImageTooLargeError)
    def image_too_large(exc):
        return _error(str(exc), 413)

    @app.errorhandler(InvalidImageError)
    def invalid_image(exc):
        return _error(str(exc), 400)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return _error(exc.description or exc.name, exc.code or 500)

    @app.errorhandler(Exception)
    def unexpected_error(exc):
        # Log the details server-side; never send stack traces or internal
        # messages to the client.
        logger.exception("Unhandled error while serving %s", request.path)
        return _error("Internal server error", 500)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("FRCS_HOST", "127.0.0.1")
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    create_app().run(host=host, port=port, debug=debug_mode)
