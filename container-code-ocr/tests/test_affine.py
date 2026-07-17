from __future__ import annotations

import cv2
import numpy as np

from container_ocr.affine import AffinePrototypeRecognizer, affine_normalize


def test_affine_normalize_centers_and_deskews_foreground() -> None:
    image = np.zeros((50, 30), dtype=np.uint8)
    cv2.line(image, (8, 44), (20, 5), 255, 4)

    stages = affine_normalize(image, 20, 30)
    moments = cv2.moments(stages.normalized, binaryImage=True)

    assert stages.normalized.shape == (30, 20)
    assert abs(stages.skew) > 0.01
    assert abs(moments["m10"] / moments["m00"] - 9.5) < 1.0
    assert abs(moments["m01"] / moments["m00"] - 14.5) < 1.0
    assert stages.matrix.shape == (2, 3)


def test_affine_prototype_recognizer_uses_nearest_class_mean() -> None:
    vertical = np.zeros((30, 20), dtype=np.uint8)
    horizontal = vertical.copy()
    vertical[3:27, 8:12] = 255
    horizontal[13:17, 2:18] = 255
    samples = [
        affine_normalize(image).normalized.reshape(-1) / 255.0
        for image in (vertical, horizontal)
    ]
    recognizer = AffinePrototypeRecognizer(np.asarray(samples), ["I", "-"], 20, 30)

    assert recognizer.predict([vertical, horizontal]) == "I-"
