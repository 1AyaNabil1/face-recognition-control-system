from flask import Flask, request, jsonify
from flask_cors import CORS
from api.recognition_service import RecognitionService
from api.utils import decode_base64_image, encode_image_base64
from api.health import health_bp

app = Flask(__name__)
CORS(app)

recognizer = RecognitionService()


@app.route("/api/recognize", methods=["POST"])
def recognize_face():
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image provided"}), 400

    image_b64 = data["image"]
    image = decode_base64_image(image_b64)

    name, score, top_matches, annotated_img = recognizer.recognize_image(image)

    response = {
        "result": name,
        "confidence": round(score, 4),
        "top_matches": [
            {"name": label, "score": round(s, 4)} for label, s in top_matches
        ],
        "annotated_image": encode_image_base64(annotated_img),
    }

    return jsonify(response), 200


if __name__ == "__main__":
    app.register_blueprint(health_bp)
    app.run(host="0.0.0.0", port=5000, debug=False)
