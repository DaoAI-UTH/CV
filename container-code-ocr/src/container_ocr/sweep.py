from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import cv2

from container_ocr.config import ensure_dir, load_config
from container_ocr.pipeline import ContainerCodePipeline
from container_ocr.viz import save_sweep_panel


SWEEPS = {
    "gaussian_kernel": [3, 5, 9],
    "adaptive_c": [3, 9, 15],
    "canny_low": [30, 60, 100],
    "morph_kernel": [9, 17, 31],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate parameter sweep panels for the lab report.")
    parser.add_argument("image")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="outputs/sweeps")
    args = parser.parse_args()

    base_config = load_config(args.config)
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    output_dir = ensure_dir(args.output_dir)
    for parameter, values in SWEEPS.items():
        panels = []
        for value in values:
            config = deepcopy(base_config)
            config["preprocess"][parameter] = value
            pipeline = ContainerCodePipeline(config)
            stages = pipeline.preprocess(image)
            stage_name = "edges" if parameter == "canny_low" else "binary"
            panels.append((f"{parameter}={value}", stages[stage_name]))

        save_sweep_panel(panels, output_dir / f"{Path(args.image).stem}_{parameter}.png", parameter)
        print(f"saved {parameter}")


if __name__ == "__main__":
    main()
