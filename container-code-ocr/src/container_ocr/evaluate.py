from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from container_ocr.config import ensure_dir, load_config
from container_ocr.pipeline import ContainerCodePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate region detection on YOLO annotations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--images", default="data/raw/test/images")
    parser.add_argument("--labels", default="data/raw/test/labels")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="outputs/eval/metrics.json")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = ContainerCodePipeline(config)
    image_paths = sorted(Path(args.images).glob("*.jpg")) + sorted(Path(args.images).glob("*.png"))

    rows = []
    for image_path in tqdm(image_paths, desc="Evaluating"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        label_path = Path(args.labels) / f"{image_path.stem}.txt"
        gt_boxes = read_yolo_boxes(label_path, image.shape[1], image.shape[0])
        prediction = pipeline.process_image(image_path)
        pred_boxes = [item.bbox for item in prediction["detections"]]
        matches, mean_iou = match_boxes(pred_boxes, gt_boxes, args.iou_threshold)
        rows.append(
            {
                "image": image_path.name,
                "gt": len(gt_boxes),
                "pred": len(pred_boxes),
                "tp": matches,
                "fp": max(0, len(pred_boxes) - matches),
                "fn": max(0, len(gt_boxes) - matches),
                "mean_iou": mean_iou,
            }
        )

    df = pd.DataFrame(rows)
    totals = df[["tp", "fp", "fn"]].sum() if not df.empty else pd.Series({"tp": 0, "fp": 0, "fn": 0})
    precision = safe_div(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_div(totals["tp"], totals["tp"] + totals["fn"])
    f1 = safe_div(2 * precision * recall, precision + recall)
    metrics = {
        "images": len(rows),
        "iou_threshold": args.iou_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": float(df["mean_iou"].mean()) if not df.empty else 0.0,
    }

    output = Path(args.output)
    ensure_dir(output.parent)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    df.to_csv(output.with_suffix(".csv"), index=False)
    print(json.dumps(metrics, indent=2))


def read_yolo_boxes(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    if not path.exists():
        return []

    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        x = int((cx - bw / 2) * width)
        y = int((cy - bh / 2) * height)
        boxes.append((x, y, int(bw * width), int(bh * height)))
    return boxes


def match_boxes(
    pred_boxes: list[tuple[int, int, int, int]],
    gt_boxes: list[tuple[int, int, int, int]],
    threshold: float,
) -> tuple[int, float]:
    used_gt, ious = set(), []
    for pred in pred_boxes:
        candidates = [(idx, iou(pred, gt)) for idx, gt in enumerate(gt_boxes) if idx not in used_gt]
        if not candidates:
            continue
        best_idx, best_iou = max(candidates, key=lambda item: item[1])
        if best_iou >= threshold:
            used_gt.add(best_idx)
            ious.append(best_iou)
    return len(ious), float(np.mean(ious)) if ious else 0.0


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return safe_div(inter, union)


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


if __name__ == "__main__":
    main()
