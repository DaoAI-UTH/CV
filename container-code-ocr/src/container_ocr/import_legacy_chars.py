from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from container_ocr.build_char_dataset import normalize_character
from container_ocr.config import ensure_dir, load_config


def augmentations(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    kernel = np.ones((2, 2), dtype=np.uint8)
    height, width = image.shape
    center = (width / 2, height / 2)
    variants = [("base", image)]
    variants.append(("erode", cv2.erode(image, kernel, iterations=1)))
    variants.append(("dilate", cv2.dilate(image, kernel, iterations=1)))
    variants.append(("blur", cv2.GaussianBlur(image, (3, 3), 0)))
    for angle in (-5, 5):
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        variants.append((f"rot{angle:+d}", rotated))
    return variants


def import_legacy(
    classifications: Path,
    flattened_images: Path,
    output: Path,
    alphabet: set[str],
    width: int,
    height: int,
    augment: bool = True,
) -> dict[str, int]:
    labels = np.loadtxt(classifications, dtype=np.float32).astype(int).ravel()
    samples = np.loadtxt(flattened_images, dtype=np.float32)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    if len(labels) != len(samples):
        raise ValueError(f"Label/sample mismatch: {len(labels)} != {len(samples)}")

    source_height, source_width = 30, 20
    if samples.shape[1] != source_height * source_width:
        raise ValueError(f"Expected 600 pixels per legacy sample, got {samples.shape[1]}")

    counts: dict[str, int] = {}
    for index, (ascii_id, pixels) in enumerate(zip(labels, samples, strict=True)):
        label = chr(int(ascii_id)).upper()
        if label not in alphabet:
            continue
        normalized = normalize_character(pixels.reshape(source_height, source_width).astype(np.uint8),
                                         width, height)
        variants = augmentations(normalized) if augment else [("base", normalized)]
        label_dir = ensure_dir(output / label)
        for variant, image in variants:
            destination = label_dir / f"legacy_{index:03d}_{variant}.png"
            if not cv2.imwrite(str(destination), image):
                raise OSError(f"Could not write {destination}")
            counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and augment GenData.py KNN characters.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", default="/home/hongdao/VIETNAMESE_LICENSE_PLATE")
    parser.add_argument("--output", default="data/processed/chars")
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    source, output = Path(args.source), ensure_dir(args.output)
    counts = import_legacy(
        source / "classifications.txt", source / "flattened_images.txt", output,
        set(config["recognition"]["alphabet"]),
        int(config["segmentation"]["char_width"]), int(config["segmentation"]["char_height"]),
        not args.no_augment,
    )
    print(f"Imported {sum(counts.values())} samples across {len(counts)} classes into {output}")
    print(" ".join(f"{label}:{counts[label]}" for label in sorted(counts)))


if __name__ == "__main__":
    main()
