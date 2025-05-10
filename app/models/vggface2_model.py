from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras.models import Model
import numpy as np


class VGGFaceModel:
    def __init__(self):
        self.model = VGGFace(
            model="vgg16",  # Switched from 'resnet50' to 'vgg16' (faster)
            include_top=False,
            input_shape=(224, 224, 3),
            pooling="avg",
        )

    def get_model(self):
        return self.model

    def preprocess(self, face):
        face = face.astype("float32")
        return preprocess_input(face, version=1)  # v1 for vgg16
