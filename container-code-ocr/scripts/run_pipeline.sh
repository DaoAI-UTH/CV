#!/usr/bin/env bash
set -euo pipefail

python -m container_ocr.cli data/raw --save-stages
