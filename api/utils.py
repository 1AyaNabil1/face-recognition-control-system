import base64
import binascii
import re
from typing import Optional

import cv2
import numpy as np

# Letters (any script), digits, spaces, underscores, apostrophes, dots and hyphens
_NAME_PATTERN = re.compile(r"^[\w .'-]+$")
MAX_NAME_LENGTH = 100


class InvalidImageError(ValueError):
    """The request did not contain a usable image."""


class ImageTooLargeError(InvalidImageError):
    """The decoded image is bigger than the configured limit."""


def decode_base64_image(b64_string: str, max_bytes: Optional[int] = None) -> np.ndarray:
    """Decode a base64 (optionally data-URL) string into a BGR image.

    Raises InvalidImageError for anything that is not a decodable image and
    ImageTooLargeError when the payload exceeds `max_bytes`.
    """
    if not isinstance(b64_string, str) or not b64_string.strip():
        raise InvalidImageError("image must be a non-empty base64 string")

    payload = b64_string.strip()
    # Accept browser data URLs such as "data:image/jpeg;base64,...."
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    # Cheap size check before decoding: base64 is ~4/3 of the raw size
    if max_bytes is not None and len(payload) * 3 // 4 > max_bytes:
        raise ImageTooLargeError(f"image is larger than {max_bytes} bytes")

    try:
        img_data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("image is not valid base64") from exc

    if max_bytes is not None and len(img_data) > max_bytes:
        raise ImageTooLargeError(f"image is larger than {max_bytes} bytes")

    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) if np_arr.size else None
    if image is None:
        raise InvalidImageError("image could not be decoded (use JPEG or PNG)")
    return image


def encode_image_base64(image):
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def normalize_person_name(name) -> str:
    """Validate a person's display name and return it stripped.

    Raises ValueError with a client-safe message when the name is unusable.
    """
    if not isinstance(name, str):
        raise ValueError("name must be a string")
    name = " ".join(name.split())
    if not name:
        raise ValueError("name must not be empty")
    if len(name) > MAX_NAME_LENGTH:
        raise ValueError(f"name must be at most {MAX_NAME_LENGTH} characters")
    if not _NAME_PATTERN.match(name):
        raise ValueError(
            "name may only contain letters, digits, spaces, "
            "apostrophes, dots, hyphens and underscores"
        )
    return name
