"""Compare training-free container-code region detectors.

Tune on validation, lock the winner, then report once on test.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

Box = tuple[int, int, int, int]


def gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def contours_to_boxes(mask: np.ndarray, image_shape: tuple[int, ...], max_candidates: int) -> list[Box]:
    height, width = image_shape[:2]
    image_area = height * width
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: list[tuple[float, Box]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_ratio = w * h / image_area
        if not 0.0005 <= area_ratio <= 0.12 or min(w, h) < 8:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if not 1.15 <= aspect <= 12:
            continue
        density = cv2.countNonZero(mask[y : y + h, x : x + w]) / (w * h)
        scored.append((density * np.log1p(w * h), (x, y, w, h)))
    scored.sort(reverse=True)
    return nms([box for _, box in scored], 0.35)[:max_candidates]


def directional_gradient(image: np.ndarray, cfg: dict) -> list[Box]:
    value = gray(image)
    blackhat = cv2.morphologyEx(
        value, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (cfg["hat"], cfg["hat"]))
    )
    masks = []
    for dx, dy, kernel_size in ((1, 0, (cfg["long"], cfg["short"])), (0, 1, (cfg["short"], cfg["long"]))):
        gradient = cv2.Scharr(blackhat, cv2.CV_32F, dx, dy)
        gradient = cv2.convertScaleAbs(gradient)
        gradient = cv2.GaussianBlur(gradient, (3, 3), 0)
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        masks.append(cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=cfg["iterations"]))
    return contours_to_boxes(cv2.bitwise_or(masks[0], masks[1]), image.shape, cfg["max_candidates"])


def morphology_text(image: np.ndarray, cfg: dict) -> list[Box]:
    value = gray(image)
    top = cv2.morphologyEx(value, cv2.MORPH_TOPHAT, np.ones((cfg["hat"], cfg["hat"]), np.uint8))
    black = cv2.morphologyEx(value, cv2.MORPH_BLACKHAT, np.ones((cfg["hat"], cfg["hat"]), np.uint8))
    enhanced = cv2.max(top, black)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((cfg["short"], cfg["long"]), np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((cfg["long"], cfg["short"]), np.uint8))
    return contours_to_boxes(cv2.bitwise_or(horizontal, vertical), image.shape, cfg["max_candidates"])


def mser_grouping(image: np.ndarray, cfg: dict) -> list[Box]:
    value = gray(image)
    detector = cv2.MSER_create(cfg["delta"], cfg["min_area"], cfg["max_area"])
    _, boxes = detector.detectRegions(value)
    component_mask = np.zeros_like(value)
    for x, y, w, h in boxes:
        aspect = w / max(h, 1)
        if 0.08 <= aspect <= 2.0 and 5 <= h <= image.shape[0] * 0.35:
            cv2.rectangle(component_mask, (x, y), (x + w, y + h), 255, -1)
    horizontal = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, np.ones((cfg["short"], cfg["long"]), np.uint8))
    vertical = cv2.morphologyEx(component_mask, cv2.MORPH_CLOSE, np.ones((cfg["long"], cfg["short"]), np.uint8))
    return contours_to_boxes(cv2.bitwise_or(horizontal, vertical), image.shape, cfg["max_candidates"])


METHODS = {"scharr_blackhat": directional_gradient, "morphology": morphology_text, "mser": mser_grouping}

GRIDS = {
    "scharr_blackhat": {"hat": [9, 15, 21], "long": [9, 15, 21], "short": [3, 5], "iterations": [1, 2], "max_candidates": [5]},
    "morphology": {"hat": [9, 15, 21], "long": [9, 15, 21], "short": [3, 5], "max_candidates": [5]},
    "mser": {"delta": [3, 5], "min_area": [10, 25], "max_area": [3000], "long": [9, 15, 21], "short": [3, 5], "max_candidates": [5]},
}


def configs(grid: dict) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values, strict=True)) for values in itertools.product(*(grid[key] for key in keys))]


def read_yolo(path: Path, width: int, height: int) -> list[Box]:
    result = []
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, w, h = map(float, parts[:5])
        result.append((int((cx - w / 2) * width), int((cy - h / 2) * height), int(w * width), int(h * height)))
    return result


def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def nms(boxes: list[Box], threshold: float) -> list[Box]:
    kept = []
    for box in boxes:
        if all(box_iou(box, other) < threshold for other in kept):
            kept.append(box)
    return kept


def evaluate(method: str, cfg: dict, split: str, root: Path) -> dict:
    tp = fp = fn = 0; matched_ious = []
    paths = sorted((root / split / "images").glob("*.jpg"))
    for path in paths:
        image = cv2.imread(str(path)); gt = read_yolo(root / split / "labels" / f"{path.stem}.txt", image.shape[1], image.shape[0])
        predictions = METHODS[method](image, cfg)
        used = set()
        for prediction in predictions:
            candidates = [(index, box_iou(prediction, box)) for index, box in enumerate(gt) if index not in used]
            if candidates:
                index, score = max(candidates, key=lambda item: item[1])
                if score >= 0.5:
                    used.add(index); tp += 1; matched_ious.append(score); continue
            fp += 1
        fn += len(gt) - len(used)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"method": method, "split": split, "config": cfg, "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0, "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0}


def save_examples(method: str, cfg: dict, root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted((root / "test" / "images").glob("*.jpg"))[:8]:
        image = cv2.imread(str(path)); canvas = image.copy()
        for x, y, w, h in METHODS[method](image, cfg):
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for x, y, w, h in read_yolo(root / "test" / "labels" / f"{path.stem}.txt", image.shape[1], image.shape[0]):
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(str(output / path.name), canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="experiments/classical_roi/results")
    args = parser.parse_args(); root = Path(args.data); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    validation = []
    for method in METHODS:
        best = max((evaluate(method, cfg, "valid", root) for cfg in configs(GRIDS[method])), key=lambda row: (row["f1"], row["mean_iou"]))
        validation.append(best); print("VALID", json.dumps(best))
    winner = max(validation, key=lambda row: (row["f1"], row["mean_iou"]))
    test = evaluate(winner["method"], winner["config"], "test", root)
    report = {"selection_rule": "highest validation F1 at IoU 0.5; test used once", "validation_winners": validation, "selected": winner, "test": test}
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_examples(winner["method"], winner["config"], root, output / "winner_examples")
    labels = [row["method"] for row in validation]; values = [row["f1"] for row in validation]
    plt.bar(labels, values); plt.ylabel("Validation F1 @ IoU 0.5"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig(output / "validation_f1.png", dpi=180); plt.close()
    print("TEST", json.dumps(test))


if __name__ == "__main__":
    main()
