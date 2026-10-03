from types import SimpleNamespace

import numpy as np
import pytest

from app.detection.yolo_detector import YOLOFaceDetector, expand_box


def yolo_box(xyxy, conf):
    return SimpleNamespace(conf=[conf], xyxy=[np.array(xyxy, dtype=float)])


class FakeYOLO:
    """Mimics the parts of an ultralytics result the detector reads."""

    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = []

    def predict(self, source, conf, iou, verbose):
        self.calls.append({"conf": conf, "iou": iou})
        return [SimpleNamespace(boxes=self.boxes)]


IMAGE = np.zeros((200, 300, 3), dtype=np.uint8)


@pytest.mark.parametrize(
    "box, margin, expected",
    [
        ((100, 50, 200, 150), 0.1, (90, 40, 210, 160)),
        ((0, 0, 100, 100), 0.5, (0, 0, 150, 150)),  # clipped at the top-left
        ((250, 150, 300, 200), 1.0, (200, 100, 300, 200)),  # clipped bottom-right
    ],
)
def test_expand_box_adds_margin_and_clips_to_the_image(box, margin, expected):
    assert expand_box(box, IMAGE.shape, margin=margin) == expected


def test_missing_weights_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="YOLO model not found"):
        YOLOFaceDetector(model_path=str(tmp_path / "missing.pt"))


@pytest.mark.parametrize(
    "box, valid",
    [
        ((10, 10, 60, 70), True),
        ((-1, 10, 60, 70), False),  # outside the image
        ((10, 10, 301, 70), False),
        ((10, 10, 30, 30), False),  # smaller than min_face_size
        ((10, 10, 150, 50), False),  # aspect ratio > 2
    ],
)
def test_validate_face_box(box, valid):
    detector = YOLOFaceDetector(model=FakeYOLO([]), min_face_size=30)
    assert detector._validate_face_box(box, IMAGE.shape) is valid


def test_detect_faces_returns_int_boxes_and_filters_bad_ones():
    model = FakeYOLO(
        [
            yolo_box([10.6, 20.2, 80.9, 100.4], 0.9),
            yolo_box([100, 20, 170, 100], 0.1),  # below confidence
            yolo_box([200, 20, 210, 30], 0.9),  # too small
        ]
    )
    detector = YOLOFaceDetector(model=model, confidence=0.35)

    boxes = detector.detect_faces(IMAGE)

    assert boxes == [(10, 20, 80, 100)]
    assert all(isinstance(v, int) for v in boxes[0])
    assert model.calls == [{"conf": 0.35, "iou": 0.5}]


def test_detect_faces_handles_empty_input():
    detector = YOLOFaceDetector(model=FakeYOLO([yolo_box([0, 0, 50, 50], 0.9)]))
    assert detector.detect_faces(None) == []
    assert detector.detect_faces(np.zeros((0, 0, 3), np.uint8)) == []


def test_falls_back_to_haar_cascade_when_yolo_finds_nothing():
    detector = YOLOFaceDetector(model=FakeYOLO([]))
    # A blank frame has no faces for either detector, and must not crash
    assert detector.detect_faces(IMAGE) == []


def test_draw_boxes_does_not_modify_the_input():
    detector = YOLOFaceDetector(model=FakeYOLO([]))
    annotated = detector.draw_boxes(IMAGE, [(10, 10, 60, 60), (100, 10, 150, 60)])
    assert annotated.any()
    assert not IMAGE.any()
