import cv2
from app.detection.yolo_detector import YOLOFaceDetector
from app.embedding.face_embedder import FaceEmbedder
from app.database.db_manager import EmbeddingDatabase
from app.recognition.face_recognizer import FaceRecognizer


class RecognitionService:
    def __init__(self):
        self.db = EmbeddingDatabase()
        self.yolo = YOLOFaceDetector(confidence=0.2)
        self.embedder = FaceEmbedder()
        self.recognizer = FaceRecognizer(embedder=self.embedder, database=self.db)

    def recognize_image(self, image):
        boxes = self.yolo.detect_faces(image)
        if not boxes:
            return "Unknown", 0.0, [], image

        x1, y1, x2, y2 = boxes[0]
        face = image[y1:y2, x1:x2]
        face = cv2.resize(face, (112, 112))

        name, score, top_matches = self.recognizer.recognize(face)
        annotated_img = self.yolo.draw_boxes(image.copy(), boxes)

        return name, score, top_matches, annotated_img
