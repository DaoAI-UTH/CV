#!/usr/bin/env bash
set -euo pipefail

python -m container_ocr.evaluate --images data/raw/test/images --labels data/raw/test/labels
