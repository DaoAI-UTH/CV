from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class AffineStages:
    """Intermediate images produced while canonicalizing one character."""

    input: np.ndarray
    binary: np.ndarray
    cropped: np.ndarray
    centered: np.ndarray
    normalized: np.ndarray
    matrix: np.ndarray
    skew: float


def affine_normalize(
    image: np.ndarray, width: int = 20, height: int = 30, margin: int = 2
) -> AffineStages:
    """Binarize, scale, center and deskew a character with an affine shear."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    points = cv2.findNonZero(binary)
    if points is None:
        empty = np.zeros((height, width), dtype=np.uint8)
        identity = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        return AffineStages(gray, binary, binary.copy(), empty, empty.copy(), identity, 0.0)

    x, y, w, h = cv2.boundingRect(points)
    cropped = binary[y : y + h, x : x + w]
    inner_w, inner_h = max(1, width - 2 * margin), max(1, height - 2 * margin)
    scale = min(inner_w / max(w, 1), inner_h / max(h, 1))
    resized_w, resized_h = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(cropped, (resized_w, resized_h), interpolation=interpolation)
    _, resized = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    centered = np.zeros((height, width), dtype=np.uint8)
    offset_x, offset_y = (width - resized_w) // 2, (height - resized_h) // 2
    centered[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized

    moments = cv2.moments(centered, binaryImage=True)
    skew = float(moments["mu11"] / moments["mu02"]) if abs(moments["mu02"]) > 1e-6 else 0.0
    skew = float(np.clip(skew, -0.7, 0.7))
    center_y = (height - 1) / 2.0
    matrix = np.array([[1.0, -skew, skew * center_y], [0.0, 1.0, 0.0]], dtype=np.float32)
    deskewed = cv2.warpAffine(
        centered,
        matrix,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    final_moments = cv2.moments(deskewed, binaryImage=True)
    if final_moments["m00"] > 0:
        cx = final_moments["m10"] / final_moments["m00"]
        cy = final_moments["m01"] / final_moments["m00"]
        tx, ty = (width - 1) / 2.0 - cx, (height - 1) / 2.0 - cy
        translation = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
        normalized = cv2.warpAffine(
            deskewed,
            translation,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        homogeneous = np.vstack([matrix, [0.0, 0.0, 1.0]])
        matrix = (np.vstack([translation, [0.0, 0.0, 1.0]]) @ homogeneous)[:2].astype(
            np.float32
        )
    else:
        normalized = deskewed

    return AffineStages(gray, binary, cropped, centered, normalized, matrix, skew)


class AffinePrototypeRecognizer:
    """Nearest class prototype after affine canonicalization; this is not KNN."""

    def __init__(self, prototypes: np.ndarray, labels: list[str], width: int, height: int):
        self.prototypes = prototypes.astype(np.float32)
        self.labels = labels
        self.width = int(width)
        self.height = int(height)

    def predict(self, images: list[np.ndarray]) -> str:
        if not images:
            return ""
        features = np.asarray(
            [
                affine_normalize(image, self.width, self.height).normalized.reshape(-1)
                / 255.0
                for image in images
            ],
            dtype=np.float32,
        )
        distances = ((features[:, None, :] - self.prototypes[None, :, :]) ** 2).mean(axis=2)
        return "".join(self.labels[index] for index in np.argmin(distances, axis=1))

    @classmethod
    def load(cls, path: str | Path) -> AffinePrototypeRecognizer | None:
        path = Path(path)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        return cls(
            data["prototypes"],
            data["labels"].tolist(),
            int(data["width"]),
            int(data["height"]),
        )
