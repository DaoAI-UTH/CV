from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from container_ocr.config import ensure_dir, load_config
from container_ocr.pipeline import ContainerCodePipeline


ALNUM = re.compile(r"^[A-Z0-9]{6,12}$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Candidate:
    source: Path
    code: str
    index: int
    image: np.ndarray

    @property
    def sample_id(self) -> str:
        return f"{self.source.stem}__{self.index:02d}"


def code_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 2 and ALNUM.fullmatch(parts[1]) else ""


def read_yolo_box(path: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    if not lines:
        return None
    parts = lines[0].split()
    if len(parts) < 5:
        return None
    _, cx, cy, box_width, box_height = map(float, parts[:5])
    x = max(0, int((cx - box_width / 2) * width))
    y = max(0, int((cy - box_height / 2) * height))
    w = min(width - x, max(1, int(box_width * width)))
    h = min(height - y, max(1, int(box_height * height)))
    return x, y, w, h


def normalize_character(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)
    ys, xs = np.where(binary > 0)
    if len(xs):
        binary = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    scale = min((width - 4) / max(binary.shape[1], 1), (height - 4) / max(binary.shape[0], 1))
    resized = cv2.resize(
        binary,
        (max(1, round(binary.shape[1] * scale)), max(1, round(binary.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.zeros((height, width), dtype=np.uint8)
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def save_sample(candidate: Candidate, label: str, output: Path, width: int, height: int) -> Path:
    destination = ensure_dir(output / label) / f"{candidate.sample_id}.png"
    normalized = normalize_character(candidate.image, width, height)
    if not cv2.imwrite(str(destination), normalized):
        raise OSError(f"Could not write {destination}")
    return destination


def iter_candidates(data: Path, split: str, pipeline: ContainerCodePipeline):
    images_dir, labels_dir = data / split / "images", data / split / "labels"
    for image_path in sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES):
        code = code_from_path(image_path)
        image = cv2.imread(str(image_path))
        if not code or image is None:
            continue
        box = read_yolo_box(labels_dir / f"{image_path.stem}.txt", image.shape[1], image.shape[0])
        if box is None:
            continue
        x, y, w, h = box
        roi = image[y : y + h, x : x + w]
        if roi.shape[0] > roi.shape[1]:
            roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
        chars = pipeline.segment_characters(roi)
        for index, char in enumerate(chars):
            yield Candidate(image_path, code, index, char), roi, len(chars) == len(code)


def append_metadata(path: Path, candidate: Candidate, label: str, automatic: bool) -> None:
    record = {"sample_id": candidate.sample_id, "source": str(candidate.source), "index": candidate.index,
              "label": label, "automatic": automatic}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            completed.add(json.loads(line)["sample_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a labeled 20x30 character dataset for KNN.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default="data")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default="data/processed/chars")
    parser.add_argument("--mode", choices=("manual", "auto"), default="manual")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    output = ensure_dir(args.output)
    metadata = output / "metadata.jsonl"
    completed = load_completed(metadata)
    pipeline = ContainerCodePipeline(config)
    alphabet = set(config["recognition"]["alphabet"])
    width = int(config["segmentation"]["char_width"])
    height = int(config["segmentation"]["char_height"])
    saved = skipped = 0

    for candidate, roi, exact_count in iter_candidates(Path(args.data), args.split, pipeline):
        if candidate.sample_id in completed:
            continue
        suggestion = candidate.code[candidate.index] if exact_count else ""
        if args.mode == "auto":
            if not suggestion:
                skipped += 1
                continue
            label = suggestion
        else:
            preview = cv2.resize(candidate.image, (200, 300), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Container ROI", roi)
            cv2.imshow("Character: press 0-9/A-Z, ENTER=accept suggestion, S=skip, Q=quit", preview)
            print(f"{candidate.sample_id}: suggestion={suggestion or '-'}", flush=True)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                skipped += 1
                continue
            label = suggestion if key in (10, 13) else chr(key).upper()
            if label not in alphabet:
                skipped += 1
                continue
        save_sample(candidate, label, output, width, height)
        append_metadata(metadata, candidate, label, args.mode == "auto")
        completed.add(candidate.sample_id)
        saved += 1
        if args.limit and saved >= args.limit:
            break

    if args.mode == "manual":
        cv2.destroyAllWindows()
    print(f"Saved {saved} characters to {output}; skipped {skipped}.")


if __name__ == "__main__":
    main()
