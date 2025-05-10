import cv2
import numpy as np


def align_face(image):
    # No-op (optional: future OpenCV eye alignment)
    return image


def preprocess_face_image(
    image, target_size=(224, 224), equalize_hist=False, apply_clahe=True
):
    image = align_face(image)
    image = cv2.resize(image, target_size)
    image = image.astype("float32")

    if equalize_hist:
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        image = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

    if apply_clahe:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l.astype(np.uint8))
        image = cv2.merge((cl, a.astype(np.uint8), b.astype(np.uint8)))
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        image = image.astype("float32")

    return image
