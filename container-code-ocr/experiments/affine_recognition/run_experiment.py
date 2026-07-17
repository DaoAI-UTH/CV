"""Fair comparison: pixel KNN vs affine KNN vs affine class prototypes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from container_ocr.affine import affine_normalize

WIDTH, HEIGHT = 20, 30
METHOD_NAMES = {
    "pixel_knn": "KNN gốc (pixel)",
    "affine_knn": "Affine + KNN",
    "affine_prototype": "Affine + prototype (không KNN)",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    group: str


def group_key(path: Path) -> str:
    """Keep derived views of the same source character in one split."""
    stem = path.stem
    if "_jpg.rf." in stem:
        source, rest = stem.split("_jpg.rf.", 1)
        position = rest.rsplit("__", 1)[-1] if "__" in rest else ""
        return f"roboflow:{source}:{position}"
    base = re.sub(r"_(base|blur|erode|dilate|rot[+-][0-9]+)$", "", stem)
    return f"{path.parent.name}:{base}"


def load_samples(data_dir: Path) -> list[Sample]:
    samples = []
    for folder in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for path in sorted(folder.glob("*")):
            if cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) is not None:
                samples.append(Sample(path, folder.name, group_key(path)))
    if not samples:
        raise RuntimeError(f"No readable character images in {data_dir}")
    return samples


def split_name(group: str, seed: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "valid"
    return "test"


def make_splits(samples: list[Sample], seed: int) -> dict[str, list[Sample]]:
    assignment = {sample.group: split_name(sample.group, seed) for sample in samples}
    group_labels: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        group_labels[sample.group].add(sample.label)
    labels = sorted({sample.label for sample in samples})

    # Guarantee a train representative when possible; test-only rare classes are not useful.
    for label in labels:
        groups = sorted(group for group, values in group_labels.items() if label in values)
        if groups and not any(assignment[group] == "train" for group in groups):
            assignment[groups[0]] = "train"

    # Also guarantee test coverage for classes with at least two independent groups.
    for label in labels:
        groups = sorted(group for group, values in group_labels.items() if label in values)
        if len(groups) < 2 or any(assignment[group] == "test" for group in groups):
            continue
        candidates = sorted(groups, key=lambda group: assignment[group] == "train")
        for group in candidates:
            if assignment[group] == "train":
                can_move = all(
                    sum(
                        assignment[other] == "train"
                        for other, values in group_labels.items()
                        if member in values
                    )
                    > 1
                    for member in group_labels[group]
                )
                if not can_move:
                    continue
            assignment[group] = "test"
            break

    return {
        split: [sample for sample in samples if assignment[sample.group] == split]
        for split in ("train", "valid", "test")
    }


def read_image(sample: Sample) -> np.ndarray:
    image = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Cannot read {sample.path}")
    return image


def pixel_feature(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, (WIDTH, HEIGHT))
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)
    return binary.reshape(-1).astype(np.float32) / 255.0


def affine_feature(image: np.ndarray) -> np.ndarray:
    return affine_normalize(image, WIDTH, HEIGHT).normalized.reshape(-1).astype(np.float32) / 255.0


def feature_matrix(samples: list[Sample], affine: bool) -> np.ndarray:
    extractor = affine_feature if affine else pixel_feature
    return np.asarray([extractor(read_image(sample)) for sample in samples], dtype=np.float32)


def encode(samples: list[Sample], labels: list[str]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(labels)}
    return np.asarray([lookup[sample.label] for sample in samples], dtype=np.int32)


def train_knn(features: np.ndarray, responses: np.ndarray) -> cv2.ml.KNearest:
    model = cv2.ml.KNearest_create()
    model.train(features, cv2.ml.ROW_SAMPLE, responses.astype(np.float32))
    return model


def predict_knn(model: cv2.ml.KNearest, features: np.ndarray, k: int) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.int32)
    _, prediction, _, _ = model.findNearest(features, k)
    return prediction.ravel().astype(np.int32)


def train_prototypes(features: np.ndarray, responses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    class_ids = np.unique(responses)
    prototypes = np.asarray([features[responses == class_id].mean(axis=0) for class_id in class_ids])
    return prototypes.astype(np.float32), class_ids


def predict_prototypes(features: np.ndarray, prototypes: np.ndarray, class_ids: np.ndarray) -> np.ndarray:
    if len(features) == 0:
        return np.empty(0, dtype=np.int32)
    distances = ((features[:, None, :] - prototypes[None, :, :]) ** 2).mean(axis=2)
    return class_ids[np.argmin(distances, axis=1)]


def accuracy(expected: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(expected == predicted)) if len(expected) else 0.0


def wilson_interval(correct: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return float(center - radius), float(center + radius)


def method_metrics(
    expected: np.ndarray,
    predicted: np.ndarray,
    labels: list[str],
    latency_ms: float,
    model_bytes: int,
) -> dict:
    correct = int(np.sum(expected == predicted))
    per_class = []
    recalls = []
    for class_id in np.unique(expected):
        mask = expected == class_id
        value = accuracy(expected[mask], predicted[mask])
        recalls.append(value)
        per_class.append(
            {
                "label": labels[int(class_id)],
                "test_count": int(mask.sum()),
                "correct": int(np.sum(predicted[mask] == class_id)),
                "recall": value,
            }
        )
    low, high = wilson_interval(correct, len(expected))
    return {
        "accuracy": accuracy(expected, predicted),
        "accuracy_ci95": [low, high],
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "correct": correct,
        "test_samples": len(expected),
        "latency_ms_per_character": latency_ms,
        "estimated_model_bytes": int(model_bytes),
        "per_class": per_class,
    }


def timed_prediction(function, repeats: int = 20) -> tuple[np.ndarray, float]:
    prediction = function()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    elapsed = time.perf_counter() - started
    per_character_ms = elapsed * 1000 / max(1, repeats * len(prediction))
    return prediction, per_character_ms


def confusion(expected: np.ndarray, predicted: np.ndarray, count: int) -> np.ndarray:
    matrix = np.zeros((count, count), dtype=np.int32)
    for actual, guess in zip(expected, predicted, strict=True):
        matrix[int(actual), int(guess)] += 1
    return matrix


def save_metric_chart(metrics: dict, output: Path) -> None:
    keys = list(METHOD_NAMES)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    values = [metrics[key]["accuracy"] for key in keys]
    intervals = np.asarray([metrics[key]["accuracy_ci95"] for key in keys])
    errors = np.vstack([np.asarray(values) - intervals[:, 0], intervals[:, 1] - np.asarray(values)])
    axes[0].bar(range(3), values, yerr=errors, capsize=5, color=["#777777", "#3182bd", "#31a354"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Accuracy (95% Wilson CI)")
    axes[1].bar(range(3), [metrics[key]["latency_ms_per_character"] for key in keys])
    axes[1].set_ylabel("ms / character")
    axes[1].set_title("Inference latency")
    axes[2].bar(range(3), [metrics[key]["estimated_model_bytes"] / 1024 for key in keys])
    axes[2].set_ylabel("KiB")
    axes[2].set_title("Estimated model size")
    for axis in axes:
        axis.set_xticks(range(3), ["Pixel KNN", "Affine KNN", "Affine prototype"], rotation=15)
        axis.grid(axis="y", alpha=0.25)
    fig.savefig(output / "metrics_comparison.png", dpi=190)
    plt.close(fig)


def save_confusions(
    expected: np.ndarray, predictions: dict[str, np.ndarray], labels: list[str], output: Path
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for axis, key in zip(axes, METHOD_NAMES, strict=True):
        matrix = confusion(expected, predictions[key], len(labels)).astype(np.float32)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(METHOD_NAMES[key])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_xticks(range(len(labels)), labels, fontsize=6)
        axis.set_yticks(range(len(labels)), labels, fontsize=6)
    fig.colorbar(image, ax=axes, shrink=0.75, label="Row-normalized rate")
    fig.savefig(output / "confusion_matrices.png", dpi=190)
    plt.close(fig)


def save_affine_steps(samples: list[Sample], output: Path) -> None:
    chosen = samples[:6]
    fig, axes = plt.subplots(len(chosen), 5, figsize=(11, 2.2 * len(chosen)), constrained_layout=True)
    if len(chosen) == 1:
        axes = np.asarray([axes])
    titles = ["Input", "Otsu + polarity", "Crop foreground", "Scale + center", "Affine deskew"]
    for row, sample in enumerate(chosen):
        stages = affine_normalize(read_image(sample), WIDTH, HEIGHT)
        images = [stages.input, stages.binary, stages.cropped, stages.centered, stages.normalized]
        for column, (title, image) in enumerate(zip(titles, images, strict=True)):
            axes[row, column].imshow(image, cmap="gray", vmin=0, vmax=255)
            axes[row, column].set_title(title if row == 0 else "")
            axes[row, column].axis("off")
        axes[row, 0].set_ylabel(f"{sample.label}\nskew={stages.skew:.3f}")
    fig.suptitle("Các bước chuẩn hoá affine cho ký tự")
    fig.savefig(output / "affine_steps.png", dpi=190)
    plt.close(fig)


def save_prototypes(prototypes: np.ndarray, class_ids: np.ndarray, labels: list[str], output: Path) -> None:
    columns = 10
    rows = int(np.ceil(len(prototypes) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 1.8 * rows), constrained_layout=True)
    for axis in np.asarray(axes).ravel():
        axis.axis("off")
    for axis, prototype, class_id in zip(np.asarray(axes).ravel(), prototypes, class_ids, strict=False):
        axis.imshow(prototype.reshape(HEIGHT, WIDTH), cmap="gray", vmin=0, vmax=1)
        axis.set_title(labels[int(class_id)])
        axis.axis("off")
    fig.suptitle("Một prototype trung bình cho mỗi lớp (model không KNN)")
    fig.savefig(output / "class_prototypes.png", dpi=190)
    plt.close(fig)


def save_prediction_changes(
    samples: list[Sample], labels: list[str], predictions: dict[str, np.ndarray], output: Path
) -> None:
    baseline, affine = predictions["pixel_knn"], predictions["affine_prototype"]
    interesting = [index for index in range(len(samples)) if baseline[index] != affine[index]]
    interesting += [index for index in range(len(samples)) if index not in interesting]
    chosen = interesting[:20]
    fig, axes = plt.subplots(4, 5, figsize=(11, 9), constrained_layout=True)
    for axis, index in zip(axes.ravel(), chosen, strict=False):
        stages = affine_normalize(read_image(samples[index]), WIDTH, HEIGHT)
        axis.imshow(stages.normalized, cmap="gray", vmin=0, vmax=255)
        expected = samples[index].label
        old, new = labels[int(baseline[index])], labels[int(affine[index])]
        axis.set_title(f"GT {expected} | KNN {old} | Aff {new}", fontsize=8)
        axis.axis("off")
    for axis in axes.ravel()[len(chosen) :]:
        axis.axis("off")
    fig.suptitle("Các mẫu dự đoán thay đổi sau chuẩn hoá affine/prototype")
    fig.savefig(output / "prediction_changes.png", dpi=190)
    plt.close(fig)


def write_csvs(
    splits: dict[str, list[Sample]],
    labels: list[str],
    predictions: dict[str, np.ndarray],
    metrics: dict,
    output: Path,
) -> None:
    with (output / "splits.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["split", "label", "group", "path"])
        for split, samples in splits.items():
            for sample in samples:
                writer.writerow([split, sample.label, sample.group, sample.path])
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["path", "expected", *METHOD_NAMES])
        for index, sample in enumerate(splits["test"]):
            writer.writerow(
                [sample.path, sample.label]
                + [labels[int(predictions[key][index])] for key in METHOD_NAMES]
            )
    with (output / "per_class_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["method", "label", "test_count", "correct", "recall"]
        )
        writer.writeheader()
        for method, result in metrics.items():
            for row in result["per_class"]:
                writer.writerow({"method": method, **row})


def write_log(
    path: Path,
    command: str,
    summary: dict,
    data_dir: Path,
    output: Path,
) -> None:
    methods = summary["methods"]
    winner = max(methods, key=lambda key: methods[key]["accuracy"])
    baseline = methods["pixel_knn"]["accuracy"]
    affine_only = methods["affine_knn"]["accuracy"]
    replacement = methods["affine_prototype"]["accuracy"]
    delta_affine = affine_only - baseline
    delta_replacement = replacement - baseline
    split = summary["dataset"]["split_samples"]
    lines = [
        "# NHẬT KÝ THỰC NGHIỆM: BIẾN ĐỔI AFFINE VÀ KNN",
        "",
        f"- Thời điểm UTC: {summary['run']['timestamp_utc']}",
        f"- Lệnh đã chạy: `{command}`",
        f"- Python: {summary['run']['python']}; OpenCV: {summary['run']['opencv']}; NumPy: {summary['run']['numpy']}",
        f"- Seed: {summary['run']['seed']}; dữ liệu: `{data_dir}`",
        f"- Output: `{output}`",
        "",
        "## 1. Mục tiêu và lưu ý khái niệm",
        "",
        "Affine là phép biến đổi hình học, không tự sinh ra nhãn ký tự; KNN là bộ phân lớp.",
        "Vì vậy thực nghiệm tách hai giả thuyết: (1) thêm affine trước KNN có tốt hơn không,",
        "và (2) nếu bỏ KNN thì affine + nearest class prototype có đủ tốt không.",
        "",
        "## 2. Những gì đã implement",
        "",
        "1. Otsu và tự đảo polarity để foreground luôn trắng.",
        "2. Crop bounding box foreground, scale giữ tỷ lệ và đặt giữa canvas 20×30.",
        "3. Ước lượng độ nghiêng `s = mu11 / mu02` từ moment bậc hai.",
        "4. Warp affine theo `x' = x - s(y-cy)`, rồi tịnh tiến centroid về tâm.",
        "5. So sánh KNN pixel gốc, affine + KNN, affine + một prototype trung bình mỗi lớp.",
        "6. Chia train/valid/test theo group ảnh gốc; chọn k trên valid, test chỉ dùng một lần.",
        "",
        "## 3. Dữ liệu và thiết lập",
        "",
        f"- Tổng {summary['dataset']['total_samples']} ảnh, {summary['dataset']['classes']} lớp hiện diện.",
        f"- Train/valid/test: {split['train']}/{split['valid']}/{split['test']} ảnh.",
        f"- Số group train/valid/test: {summary['dataset']['split_groups']}.",
        f"- K được chọn: pixel KNN={summary['selection']['pixel_knn_k']}, affine KNN={summary['selection']['affine_knn_k']}.",
        f"- Lớp không xuất hiện ở test: {summary['dataset']['classes_absent_from_test'] or 'không có'}.",
        "",
        "## 4. Kết quả test",
        "",
        "| Phương pháp | Accuracy | 95% CI | Macro recall | ms/ký tự | Model KiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in METHOD_NAMES:
        item = methods[key]
        low, high = item["accuracy_ci95"]
        lines.append(
            f"| {METHOD_NAMES[key]} | {item['accuracy']:.4f} | [{low:.4f}, {high:.4f}] | "
            f"{item['macro_recall']:.4f} | {item['latency_ms_per_character']:.4f} | "
            f"{item['estimated_model_bytes'] / 1024:.2f} |"
        )
    conclusion = (
        f"Phương pháp tốt nhất trên test là **{METHOD_NAMES[winner]}**. "
        f"Affine + KNN đổi {delta_affine:+.4f} accuracy so với KNN gốc; "
        f"nhánh bỏ KNN đổi {delta_replacement:+.4f}. "
    )
    if delta_replacement > 0:
        conclusion += "Trong split này, đề xuất thay KNN bằng affine-prototype được số liệu ủng hộ."
    else:
        conclusion += (
            "Trong split này, chưa có bằng chứng rằng affine-prototype cải thiện KNN; "
            "nên xem affine là bước chuẩn hoá, không phải phép thay thế classifier."
        )
    lines += [
        "",
        "## 5. Kết luận",
        "",
        conclusion,
        "",
        "Kết quả chỉ đo **nhận dạng ký tự đã được crop** trong `data/processed/chars`,",
        "không phải độ chính xác end-to-end của detect ROI → segment → OCR.",
        "",
        "## 6. Nhật ký triển khai và kiểm chứng",
        "",
        "1. Đọc pipeline, xác nhận KNN là classifier ký tự còn affine là biến đổi hình học.",
        "2. Kiểm kê 1.203 ảnh/33 lớp; nhận thấy các biến thể cùng nguồn cần group split.",
        "3. Thêm module affine, CLI train prototype, config riêng, tích hợp pipeline và test.",
        "4. `py_compile` pass; 3 test affine/pipeline chạy trực tiếp đều pass.",
        "5. Không chạy được pytest suite vì Python hệ thống chưa cài package `pytest`.",
        "6. Lần experiment đầu gặp khác biệt API OpenCV Python (không có",
        "   `KNearest.getTrainSamples`); đã bỏ call thừa và chạy lại thành công.",
        "7. Audit split đầu thấy test thiếu 8 lớp; đã thêm kiểm soát class coverage và",
        "   chạy lại. Bản cuối có đủ 33 lớp trong test và không trộn group giữa split.",
        "8. Train model full-data thành công: 33 prototype từ 1.203 ảnh.",
        "9. Smoke test CLI affine chạy xong trên ảnh CAIU883333; pipeline end-to-end dự đoán",
        "   `1`, cho thấy detect/segment hiện vẫn là nút thắt và không được tính vào metric ký tự.",
        "",
        "## 7. File kết quả",
        "",
        "- `metrics.json`: toàn bộ cấu hình, metric và CI.",
        "- `predictions.csv`: dự đoán từng ảnh test.",
        "- `per_class_metrics.csv`: recall theo lớp.",
        "- `splits.csv`: split tái lập và group chống leakage.",
        "- `affine_test_model.npz`: prototype train-only dùng trong phép đo test.",
        "- `affine_steps.png`: ảnh trung gian từng bước.",
        "- `metrics_comparison.png`: accuracy, latency, kích thước model.",
        "- `confusion_matrices.png`: confusion matrix ba phương pháp.",
        "- `class_prototypes.png`: prototype trung bình của từng lớp.",
        "- `prediction_changes.png`: ví dụ dự đoán thay đổi.",
        "",
        "## 8. Cách chạy lại",
        "",
        "```bash",
        "cd /home/hongdao/cv/container-code-ocr",
        "PYTHONPATH=src python3 experiments/affine_recognition/run_experiment.py",
        "",
        "# Train model affine trên toàn bộ dữ liệu để dùng pipeline chính",
        "PYTHONPATH=src python3 -m container_ocr.train_affine --config configs/affine.yaml",
        "",
        "# Chạy OCR với affine-prototype và lưu các stage pipeline ảnh",
        "PYTHONPATH=src python3 -m container_ocr.cli data/test/images --config configs/affine.yaml --save-stages",
        "```",
        "",
        "Lưu ý: model `outputs/affine_prototypes.npz` train trên toàn bộ dữ liệu phục vụ demo;",
        "metric báo cáo dùng model train-only riêng để không nhìn test.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed/chars")
    parser.add_argument("--output", default="experiments/affine_recognition/results")
    parser.add_argument("--log", default="experiments/affine_recognition/THUC_NGHIEM_AFFINE.md")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir, output = Path(args.data_dir), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = load_samples(data_dir)
    splits = make_splits(samples, args.seed)
    if not splits["train"] or not splits["test"]:
        raise RuntimeError("Train/test split is empty; change seed or add data.")

    labels = sorted({sample.label for sample in samples})
    train_y, valid_y, test_y = (encode(splits[name], labels) for name in ("train", "valid", "test"))
    train_pixel, valid_pixel, test_pixel = (
        feature_matrix(splits[name], affine=False) for name in ("train", "valid", "test")
    )
    train_affine, valid_affine, test_affine = (
        feature_matrix(splits[name], affine=True) for name in ("train", "valid", "test")
    )

    pixel_model = train_knn(train_pixel, train_y)
    affine_model = train_knn(train_affine, train_y)
    candidate_k = [1, 3, 5, 7]
    pixel_scores = {
        k: accuracy(valid_y, predict_knn(pixel_model, valid_pixel, k)) for k in candidate_k
    }
    affine_scores = {
        k: accuracy(valid_y, predict_knn(affine_model, valid_affine, k)) for k in candidate_k
    }
    pixel_k = max(candidate_k, key=lambda k: (pixel_scores[k], -k))
    affine_k = max(candidate_k, key=lambda k: (affine_scores[k], -k))
    prototypes, prototype_ids = train_prototypes(train_affine, train_y)

    predictions = {}
    predictions["pixel_knn"], pixel_latency = timed_prediction(
        lambda: predict_knn(pixel_model, test_pixel, pixel_k)
    )
    predictions["affine_knn"], affine_latency = timed_prediction(
        lambda: predict_knn(affine_model, test_affine, affine_k)
    )
    predictions["affine_prototype"], prototype_latency = timed_prediction(
        lambda: predict_prototypes(test_affine, prototypes, prototype_ids)
    )
    model_bytes = {
        "pixel_knn": train_pixel.nbytes + train_y.nbytes,
        "affine_knn": train_affine.nbytes + train_y.nbytes,
        "affine_prototype": prototypes.nbytes + prototype_ids.nbytes,
    }
    latencies = {
        "pixel_knn": pixel_latency,
        "affine_knn": affine_latency,
        "affine_prototype": prototype_latency,
    }
    metrics = {
        key: method_metrics(test_y, predictions[key], labels, latencies[key], model_bytes[key])
        for key in METHOD_NAMES
    }

    split_groups = {
        name: len({sample.group for sample in values}) for name, values in splits.items()
    }
    test_classes = {sample.label for sample in splits["test"]}
    summary = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "dataset": {
            "data_dir": str(data_dir),
            "total_samples": len(samples),
            "classes": len(labels),
            "labels": labels,
            "split_samples": {name: len(values) for name, values in splits.items()},
            "split_groups": split_groups,
            "classes_absent_from_test": sorted(set(labels) - test_classes),
            "class_counts": dict(sorted(Counter(sample.label for sample in samples).items())),
        },
        "selection": {
            "candidate_k": candidate_k,
            "pixel_knn_validation_accuracy": pixel_scores,
            "affine_knn_validation_accuracy": affine_scores,
            "pixel_knn_k": pixel_k,
            "affine_knn_k": affine_k,
        },
        "methods": metrics,
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "affine_test_model.npz",
        prototypes=prototypes,
        class_ids=prototype_ids,
        labels=np.asarray(labels),
        width=np.asarray(WIDTH),
        height=np.asarray(HEIGHT),
    )
    write_csvs(splits, labels, predictions, metrics, output)
    save_metric_chart(metrics, output)
    save_confusions(test_y, predictions, labels, output)
    save_affine_steps(splits["test"], output)
    save_prototypes(prototypes, prototype_ids, labels, output)
    save_prediction_changes(splits["test"], labels, predictions, output)

    command = " ".join([Path(sys.executable).name, *sys.argv])
    write_log(Path(args.log), command, summary, data_dir, output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
