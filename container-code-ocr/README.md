# Container Code OCR

Classical computer vision pipeline for detecting and recognizing shipping container
codes from real images. The project is designed for the Image Processing and Computer
Vision final assignment: it uses preprocessing, edge/region detection, segmentation,
and image recognition, then reports intermediate images, parameter sweeps, and metrics.

## Dataset

Recommended dataset: **Container Character Codes** on Roboflow Universe.

- URL: https://universe.roboflow.com/public-workspace-n6wxn/container-character-codes
- Task: object detection for container code regions
- License shown by Roboflow: CC BY 4.0
- Export format to use: YOLOv8 or YOLOv5

Place the exported dataset like this:

```text
data/raw/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
```

For character recognition, create segmented character folders:

```text
data/processed/chars/
  0/*.png
  1/*.png
  ...
  A/*.png
  B/*.png
  ...
  Z/*.png
```

You can build these folders manually from the detected crops or use the pipeline output
as a starting point. Keep a small validation subset with known labels for the report.

## Project Layout

```text
container-code-ocr/
  configs/              # YAML parameters for the pipeline
  data/                 # raw and processed data, ignored by git
  docs/                 # assignment notes and report guidance
  notebooks/            # optional notebooks for final presentation
  outputs/              # generated stages, sweeps, predictions, metrics
  reports/              # report draft and figures
  scripts/              # short reproducible commands
  src/container_ocr/    # clean source package
  tests/                # smoke tests
```

## Setup

```bash
cd /home/hongdao/cv/container-code-ocr
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run Detection and OCR

```bash
python -m container_ocr.cli data/raw/test/images --save-stages
```

Outputs:

- `outputs/*_result.png`: final image with detected region and predicted text
- `outputs/intermediate/*_gray.png`
- `outputs/intermediate/*_contrast.png`
- `outputs/intermediate/*_blur.png`
- `outputs/intermediate/*_binary.png`
- `outputs/intermediate/*_edges.png`

If `outputs/knn_chars.npz` does not exist, the pipeline still detects code regions and
segments characters, but it will not print recognized text.

## Train Character KNN

```bash
python -m container_ocr.train_knn \
  --data-dir data/processed/chars \
  --output outputs/knn_chars.npz
```

The KNN model expects one folder per character. This mirrors the reference report's
KNN logic while changing the problem domain from vehicle plates to container codes.

## Parameter Sweep

The assignment requires at least three values for important parameters. Generate panels:

```bash
python -m container_ocr.sweep data/raw/test/images/example.jpg
```

Current sweeps:

- `gaussian_kernel`: noise reduction from Chapter 2
- `adaptive_c`: binarization threshold behavior from Chapter 2/4
- `canny_low`: edge detection from Chapter 3
- `morph_kernel`: morphology and contrast enhancement from Chapter 2/4

## Evaluation

Evaluate detection against YOLO labels:

```bash
python -m container_ocr.evaluate \
  --images data/raw/test/images \
  --labels data/raw/test/labels
```

Metrics are saved to:

- `outputs/eval/metrics.json`
- `outputs/eval/metrics.csv`

Reported metrics:

- precision
- recall
- F1
- mean IoU

## Assignment Mapping

Goal statement:

> We detect and recognize shipping container identification codes from real-world
> container images. We hypothesize that contrast enhancement plus adaptive thresholding
> and Canny-based contour filtering can localize code regions well enough for KNN-based
> character recognition. Success is measured by detection precision/recall at IoU 0.5,
> mean IoU, and character/text recognition accuracy on a manually labeled subset.

Techniques used:

- Chapter 2: grayscale/HSV conversion, Top Hat/Black Hat morphology, Gaussian filtering
- Chapter 3: Canny edge detection and contour-based region proposal
- Chapter 4: threshold-based segmentation of characters
- Chapter 5: KNN character recognition

At least two required techniques are from Chapters 3, 4, or 5.

## Notes

This project intentionally keeps the original OCR pipeline idea but changes the dataset
and application domain. That reduces implementation risk while avoiding a near-copy of
the license plate report.
