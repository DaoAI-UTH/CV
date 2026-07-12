from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from container_ocr.config import ensure_dir


def draw_results(image: np.ndarray, results: list[dict]) -> np.ndarray:
    canvas = image.copy()
    for result in results:
        x, y, w, h = result["bbox"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 220, 0), 2)
        label = result["text"] or "code-region"
        cv2.putText(canvas, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
    return canvas


def save_stages(stages: dict[str, np.ndarray], out_dir: str | Path, stem: str) -> None:
    out_dir = ensure_dir(out_dir)
    for name, image in stages.items():
        cv2.imwrite(str(out_dir / f"{stem}_{name}.png"), image)


def save_sweep_panel(images: list[tuple[str, np.ndarray]], out_path: str | Path, title: str) -> None:
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4), constrained_layout=True)
    if cols == 1:
        axes = [axes]

    for axis, (label, image) in zip(axes, images, strict=True):
        axis.imshow(image, cmap="gray" if image.ndim == 2 else None)
        axis.set_title(label)
        axis.axis("off")

    fig.suptitle(title)
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
