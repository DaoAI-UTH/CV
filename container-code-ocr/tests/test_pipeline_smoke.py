from __future__ import annotations

import numpy as np

from container_ocr.config import load_config
from container_ocr.pipeline import ContainerCodePipeline


def test_preprocess_returns_required_stages() -> None:
    config = load_config("configs/default.yaml")
    pipeline = ContainerCodePipeline(config)
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    image[45:75, 60:180] = 255

    stages = pipeline.preprocess(image)

    assert set(stages) == {"gray", "contrast", "blur", "binary", "edges"}
    assert stages["binary"].shape == image.shape[:2]
