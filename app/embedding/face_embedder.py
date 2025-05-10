import insightface
import cv2
import numpy as np


class FaceEmbedder:
    def __init__(self):
        self.model = insightface.app.FaceAnalysis(
            name="buffalo_l", providers=["CPUExecutionProvider"]
        )
        self.model.prepare(ctx_id=0)

    def get_embedding(self, face):
        # Resize to InsightFace standard size
        face_resized = cv2.resize(face, (112, 112))
        faces = self.model.get(face_resized)
        if faces and len(faces) > 0:
            return faces[0].embedding
        return np.zeros((512,))
