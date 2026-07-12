#!/usr/bin/env bash
set -euo pipefail

python -m container_ocr.train_knn --data-dir data/processed/chars --output outputs/knn_chars.npz
