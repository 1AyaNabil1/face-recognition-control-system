import sys
import os
import modal
from flask import Flask, request, jsonify

# Add root directory for imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

app = modal.App("face-recognition-system")

# Define the image with necessary dependencies, including FastAPI
image = (
    modal.Image.debian_slim()
    .apt_install(
        "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxext6", "libxrender-dev"
    )
    .pip_install(
        "flask==2.3.3",
        "flask-cors==4.0.1",
        "onnxruntime==1.16.0",
        "insightface==0.7.3",
        "numpy==1.26.4",
        "pillow==10.4.0",
        "scikit-learn==1.5.1",
        "gunicorn==23.0.0",
        "ultralytics==8.2.58",
        "opencv-python-headless==4.10.0.84",
        "fastapi[standard]==0.111.0",  # Added FastAPI with standard extensions
    )
)

flask_app = Flask(__name__)


@app.function(image=image, timeout=600)
@modal.web_endpoint(method="POST")
def recognize_face(data: dict):
    from app.recognition.face_recognizer import FaceRecognizer
    from api.utils import decode_base64_image, encode_image_base64

    recognizer = FaceRecognizer()

    try:
        if not data or "image" not in data:
            return {"error": "No image provided"}, 400

        image_b64 = data["image"]
        image = decode_base64_image(image_b64)
        name, score, top_matches, annotated_img = recognizer.recognize_image(image)

        response = {
            "result": name,
            "confidence": round(float(score), 4),
            "top_matches": [
                {"name": label, "score": round(float(s), 4)} for label, s in top_matches
            ],
            "annotated_image": encode_image_base64(annotated_img),
        }
        return response, 200

    except Exception as e:
        return {"error": str(e)}, 500
