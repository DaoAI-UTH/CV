.PHONY: install run train evaluate sweep test

install:
	python3 -m pip install -e .

run:
	python3 -m container_ocr.cli data/raw/test/images --save-stages

train:
	python3 -m container_ocr.train_knn --data-dir data/processed/chars --output outputs/knn_chars.npz

evaluate:
	python3 -m container_ocr.evaluate --images data/raw/test/images --labels data/raw/test/labels

sweep:
	python3 -m container_ocr.sweep $(IMAGE)

test:
	python3 -m pytest tests
