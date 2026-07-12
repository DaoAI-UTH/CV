from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from container_ocr.config import ensure_dir, load_config
from container_ocr.pipeline import ContainerCodePipeline
from container_ocr.viz import draw_results, save_stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and recognize shipping container codes.")
    parser.add_argument("images", nargs="+", help="Image files or directories.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--save-stages", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = ensure_dir(args.output_dir or config["paths"]["output_dir"])
    pipeline = ContainerCodePipeline(config)

    image_paths = []
    for item in args.images:
        path = Path(item)
        if path.is_dir():
            image_paths.extend(sorted(path.glob("*.jpg")))
            image_paths.extend(sorted(path.glob("*.jpeg")))
            image_paths.extend(sorted(path.glob("*.png")))
        else:
            image_paths.append(path)

    for image_path in image_paths:
        result = pipeline.process_image(image_path)
        rendered = draw_results(result["image"], result["results"])
        cv2.imwrite(str(output_dir / f"{image_path.stem}_result.png"), rendered)
        if args.save_stages:
            save_stages(result["stages"], output_dir / "intermediate", image_path.stem)
        text = result["results"][0]["text"] if result["results"] else ""
        print(f"{image_path}: {text or 'no-code-found'}")


if __name__ == "__main__":
    main()
