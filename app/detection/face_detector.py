# app/detection/face_detector.py
import cv2


class FaceDetector:
    def __init__(self, target_size=(224, 224)):
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.target_size = target_size

    def extract_face(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return None

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        return cv2.resize(face, self.target_size)
