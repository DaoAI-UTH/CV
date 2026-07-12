from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from container_ocr.config import ensure_dir, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train OpenCV KNN on segmented character folders.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default=None, help="Folder layout: data_dir/A/*.png, data_dir/0/*.png, ...")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir or config["paths"]["char_train_dir"])
    output = Path(args.output or config["paths"]["model"])
    width = int(config["segmentation"]["char_width"])
    height = int(config["segmentation"]["char_height"])
    alphabet = list(config["recognition"]["alphabet"])

    samples, responses, labels = [], [], alphabet
    for label_id, label in enumerate(labels):
        folder = data_dir / label
        for image_path in sorted(folder.glob("*")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (width, height))
            _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            samples.append(image.reshape(-1).astype(np.float32) / 255.0)
            responses.append(label_id)

    if not samples:
        raise SystemExit(f"No character images found in {data_dir}. Expected subfolders named 0-9/A-Z.")

    ensure_dir(output.parent)
    np.savez_compressed(
        output,
        samples=np.array(samples, dtype=np.float32),
        responses=np.array(responses, dtype=np.float32),
        labels=np.array(labels),
    )
    print(f"Saved {len(samples)} samples to {output}")


if __name__ == "__main__":
    main()
