from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from container_ocr.affine import affine_normalize
from container_ocr.config import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train affine-normalized class prototypes.")
    parser.add_argument("--config", default="configs/affine.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir or config["paths"]["char_train_dir"])
    output = Path(args.output or config["paths"]["model"])
    width = int(config["segmentation"]["char_width"])
    height = int(config["segmentation"]["char_height"])

    labels, prototypes, counts = [], [], []
    for label in config["recognition"]["alphabet"]:
        features = []
        for path in sorted((data_dir / label).glob("*")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                normalized = affine_normalize(image, width, height).normalized
                features.append(normalized.reshape(-1).astype(np.float32) / 255.0)
        if features:
            labels.append(label)
            prototypes.append(np.mean(features, axis=0))
            counts.append(len(features))

    if not prototypes:
        raise SystemExit(f"No character images found in {data_dir}.")
    ensure_dir(output.parent)
    np.savez_compressed(
        output,
        prototypes=np.asarray(prototypes, dtype=np.float32),
        labels=np.asarray(labels),
        counts=np.asarray(counts, dtype=np.int32),
        width=np.asarray(width),
        height=np.asarray(height),
    )
    print(f"Saved {len(prototypes)} class prototypes from {sum(counts)} images to {output}")


if __name__ == "__main__":
    main()
