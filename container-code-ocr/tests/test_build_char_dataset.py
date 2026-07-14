from pathlib import Path

import cv2
import numpy as np

from container_ocr.build_char_dataset import Candidate, code_from_path, normalize_character, save_sample


def test_code_from_roboflow_filename() -> None:
    path = Path("01012021050110310403_CAIU883333_jpg.rf.hash.jpg")
    assert code_from_path(path) == "CAIU883333"


def test_normalize_character_centers_foreground() -> None:
    image = np.zeros((40, 12), dtype=np.uint8)
    image[3:37, 4:8] = 255
    normalized = normalize_character(image, 20, 30)
    assert normalized.shape == (30, 20)
    assert normalized.dtype == np.uint8
    assert 0 < cv2.countNonZero(normalized) < normalized.size / 2


def test_save_sample_uses_label_folder(tmp_path: Path) -> None:
    image = np.zeros((30, 20), dtype=np.uint8)
    image[4:26, 8:12] = 255
    candidate = Candidate(Path("source.jpg"), "A12345", 0, image)
    saved = save_sample(candidate, "A", tmp_path, 20, 30)
    assert saved == tmp_path / "A" / "source__00.png"
    assert cv2.imread(str(saved), cv2.IMREAD_GRAYSCALE).shape == (30, 20)
