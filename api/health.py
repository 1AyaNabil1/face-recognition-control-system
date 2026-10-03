from flask import Blueprint, current_app, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/api/health", methods=["GET"])
def health_check():
    state = current_app.extensions.get("frcs", {})
    return jsonify(
        {
            "status": "OK",
            "message": "API is running",
            # Models load lazily on the first recognition/enrollment request
            "models_loaded": state.get("service") is not None,
        }
    ), 200
