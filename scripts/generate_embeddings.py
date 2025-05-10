import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.detection.yolo_detector import YOLOFaceDetector
from app.embedding.face_embedder import FaceEmbedder
from app.database.db_manager import EmbeddingDatabase


def generate_embeddings(dataset_path="dataset"):
    db_path = "app/database/embeddings.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print("[🗑️] Old embeddings.db removed. Starting fresh!")

    db = EmbeddingDatabase()
    db.initialize()

    detector = YOLOFaceDetector(confidence=0.2)
    embedder = FaceEmbedder()

    os.makedirs("debug_crops", exist_ok=True)

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[!] Skipping unreadable file: {img_path}")
                continue

            boxes = detector.detect_faces(image)
            if not boxes:
                print(f"[!] No face found in {img_path}")
                continue

            x1, y1, x2, y2 = boxes[0]
            face = image[y1:y2, x1:x2]
            face = cv2.resize(face, (224, 224))

            # Save the cropped face for manual inspection
            cv2.imwrite(f"debug_crops/{person_name}_{img_name}", face)

            embedding = embedder.get_embedding(face)
            if embedding.shape != (512,):
                print(
                    f"[!] Skipping corrupted embedding for {person_name}: {img_name} (shape: {embedding.shape})"
                )
                continue

            db.insert_embedding(person_name, embedding.tolist(), img_path)
            print(f"[✔] Embedded {person_name}: {img_name}")


if __name__ == "__main__":
    generate_embeddings()
