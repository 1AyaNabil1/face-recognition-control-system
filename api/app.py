from flask import Flask, request, jsonify
from flask_cors import CORS
from app.recognition.face_recognizer import FaceRecognizer
from api.utils import decode_base64_image, encode_image_base64
import os

app = Flask(__name__)
CORS(app)

recognizer = FaceRecognizer()


@app.route("/api/recognize", methods=["POST"])
def recognize_face():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image provided"}), 400
    try:
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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/add-person", methods=["POST"])
def add_person():
    data = request.get_json()
    if not data or "name" not in data or "image" not in data:
        return jsonify({"error": "Name or image missing"}), 400
    try:
        image_b64 = data["image"]
        name = data["name"]
        image = decode_base64_image(image_b64)
        recognizer.add_new_person(image, name)
        return jsonify({"message": "Person added successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("DEBUG", "False").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
