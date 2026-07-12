from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    score: float
    crop: np.ndarray


class ContainerCodePipeline:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.knn, self.labels = self._load_knn(config["paths"]["model"])

    def process_image(self, image_path: str | Path) -> dict[str, Any]:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        stages = self.preprocess(image)
        detections = self.detect_regions(image, stages["edges"], stages["binary"])
        results = []
        for detection in detections:
            chars = self.segment_characters(detection.crop)
            text = self.recognize(chars)
            results.append({"bbox": detection.bbox, "score": detection.score, "text": text, "chars": chars})

        return {"image": image, "stages": stages, "detections": detections, "results": results}

    def preprocess(self, image: np.ndarray) -> dict[str, np.ndarray]:
        p = self.config["preprocess"]
        gray = self._to_gray(image, p["hsv_value_channel"])
        k = int(p["morph_kernel"])
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        contrast = cv2.add(cv2.subtract(gray, black_hat), top_hat)

        g = self._odd(p["gaussian_kernel"])
        blur = cv2.GaussianBlur(contrast, (g, g), 0)
        block = max(3, self._odd(p["adaptive_block"]))
        binary = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            int(p["adaptive_c"]),
        )
        edges = cv2.Canny(binary, int(p["canny_low"]), int(p["canny_high"]))
        return {"gray": gray, "contrast": contrast, "blur": blur, "binary": binary, "edges": edges}

    def detect_regions(
        self, image: np.ndarray, edges: np.ndarray, binary: np.ndarray | None = None
    ) -> list[Detection]:
        d = self.config["detection"]
        img_area = image.shape[0] * image.shape[1]
        source = cv2.bitwise_or(edges, binary) if binary is not None else edges
        long_side = self._odd(d.get("close_kernel_long", 35))
        short_side = self._odd(d.get("close_kernel_short", 5))
        iterations = int(d.get("close_iterations", 2))
        proposal_masks = []
        for size in ((long_side, short_side), (short_side, long_side)):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
            proposal_masks.append(
                cv2.morphologyEx(source, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            )

        contours = []
        for mask in proposal_masks:
            found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours.extend(found)
        detections = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if not d["min_area_ratio"] * img_area <= area <= d["max_area_ratio"] * img_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = max(w / max(h, 1), h / max(w, 1))
            if not d["min_aspect"] <= aspect <= d["max_aspect"]:
                continue

            crop = self._crop_with_padding(image, x, y, w, h, int(d["padding"]))
            foreground_density = cv2.countNonZero(source[y : y + h, x : x + w]) / max(w * h, 1)
            rectangularity = area / max(w * h, 1)
            score = float((0.5 * rectangularity + 0.5 * foreground_density) * np.log1p(area))
            detections.append(Detection((x, y, w, h), score, crop))

        detections.sort(key=lambda item: item.score, reverse=True)
        return self._non_max_suppression(detections, 0.35)[: int(d["max_candidates"])]

    def segment_characters(self, crop: np.ndarray) -> list[np.ndarray]:
        s = self.config["segmentation"]
        stages = self.preprocess(crop)
        binary = stages["binary"]
        h, w = binary.shape
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area_ratio = (bw * bh) / max(w * h, 1)
            height_ratio = bh / max(h, 1)
            aspect = bw / max(bh, 1)
            if area_ratio < s["min_char_area_ratio"]:
                continue
            if not s["min_char_height_ratio"] <= height_ratio <= s["max_char_height_ratio"]:
                continue
            if not s["min_char_aspect"] <= aspect <= s["max_char_aspect"]:
                continue
            boxes.append((x, y, bw, bh))

        if not boxes:
            return []

        vertical = crop.shape[0] > crop.shape[1] * 1.2
        boxes.sort(key=lambda box: (box[1], box[0]) if vertical else (box[0], box[1]))
        chars = []
        for x, y, bw, bh in boxes:
            char = binary[y : y + bh, x : x + bw]
            char = cv2.resize(char, (int(s["char_width"]), int(s["char_height"])))
            chars.append(char)
        return chars

    def recognize(self, chars: list[np.ndarray]) -> str:
        if self.knn is None or not chars:
            return ""

        samples = np.array([char.reshape(-1).astype(np.float32) / 255.0 for char in chars])
        _, results, _, _ = self.knn.findNearest(samples, int(self.config["recognition"]["k"]))
        return "".join(self.labels[int(label)] for label in results.ravel())

    @staticmethod
    def _to_gray(image: np.ndarray, use_hsv_value: bool) -> np.ndarray:
        if use_hsv_value:
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _crop_with_padding(image: np.ndarray, x: int, y: int, w: int, h: int, pad: int) -> np.ndarray:
        y1, y2 = max(0, y - pad), min(image.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(image.shape[1], x + w + pad)
        return image[y1:y2, x1:x2]

    @staticmethod
    def _odd(value: int) -> int:
        value = int(value)
        return value if value % 2 else value + 1

    @staticmethod
    def _non_max_suppression(detections: list[Detection], threshold: float) -> list[Detection]:
        kept = []
        for candidate in detections:
            if all(ContainerCodePipeline._bbox_iou(candidate.bbox, item.bbox) < threshold for item in kept):
                kept.append(candidate)
        return kept

    @staticmethod
    def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = aw * ah + bw * bh - intersection
        return intersection / union if union else 0.0

    @staticmethod
    def _load_knn(path: str | Path):
        path = Path(path)
        if not path.exists():
            return None, []

        data = np.load(path, allow_pickle=True)
        knn = cv2.ml.KNearest_create()
        knn.train(data["samples"].astype(np.float32), cv2.ml.ROW_SAMPLE, data["responses"].astype(np.float32))
        return knn, data["labels"].tolist()
