"""YOLO ROI -> classical segmentation -> HOG-KNN OCR experiment."""
from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

ALNUM = re.compile(r"^[A-Z0-9]{6,12}$")


def code_from_path(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[1] if len(parts) > 2 and ALNUM.fullmatch(parts[1]) else ""


def read_box(label: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
    if not lines:
        return None
    _, cx, cy, bw, bh = map(float, lines[0].split()[:5])
    return int((cx - bw / 2) * width), int((cy - bh / 2) * height), int(bw * width), int(bh * height)


def crop(image: np.ndarray, box: tuple[int, int, int, int], padding: float = 0.08) -> np.ndarray:
    x, y, w, h = box; px, py = int(w * padding), int(h * padding)
    return image[max(0, y-py):min(image.shape[0], y+h+py), max(0, x-px):min(image.shape[1], x+w+px)]


def normalize_roi(roi: np.ndarray, target_height: int) -> tuple[np.ndarray, bool]:
    rotated = roi.shape[0] > roi.shape[1]
    if rotated:
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    scale = target_height / max(roi.shape[0], 1)
    resized = cv2.resize(roi, (max(1, int(roi.shape[1] * scale)), target_height), interpolation=cv2.INTER_CUBIC)
    return resized, rotated


def preprocess(roi: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    normalized, _ = normalize_roi(roi, cfg["target_height"])
    gray = cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) if cfg["clahe"] else gray
    blur = cv2.GaussianBlur(contrast, (cfg["blur"], cfg["blur"]), 0) if cfg["blur"] > 1 else contrast
    if cfg["threshold"] == "otsu":
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, cfg["block"], cfg["c"])
    if cv2.countNonZero(binary) / binary.size > 0.5:
        binary = cv2.bitwise_not(binary)
    k = cfg["morph"]
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) if k > 1 else binary
    return {"roi": roi, "normalized": normalized, "gray": gray, "contrast": contrast, "blur": blur, "binary": binary, "cleaned": cleaned}


def segment(stages: dict[str, np.ndarray]) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    binary = stages["cleaned"]; height, width = binary.shape
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    boxes = []
    for x, y, w, h, area in stats[1:count]:
        hr, wr = h / height, w / width
        if 0.28 <= hr <= 0.98 and 0.006 <= wr <= 0.22 and 0.08 <= w / max(h, 1) <= 1.25 and area >= 20:
            boxes.append((int(x), int(y), int(w), int(h)))
    if boxes:
        median_y = float(np.median([y + h / 2 for _, y, _, h in boxes]))
        median_h = float(np.median([h for _, _, _, h in boxes]))
        boxes = [box for box in boxes if abs((box[1] + box[3] / 2) - median_y) <= max(median_h * 0.65, 8)]
    boxes.sort(key=lambda box: box[0])
    chars = [binary[y:y+h, x:x+w] for x, y, w, h in boxes]
    return chars, boxes


def configurations() -> list[dict]:
    result = []
    for target, clahe, blur, threshold, morph in itertools.product([128, 192], [False, True], [1, 3], ["otsu", "adaptive"], [1, 2]):
        for block, c in ([(19, 7), (31, 11)] if threshold == "adaptive" else [(0, 0)]):
            result.append({"target_height": target, "clahe": clahe, "blur": blur, "threshold": threshold, "block": block, "c": c, "morph": morph})
    return result


def split_samples(root: Path, split: str):
    for path in sorted((root/split/"images").glob("*.jpg")):
        code = code_from_path(path)
        image = cv2.imread(str(path)); box = read_box(root/split/"labels"/f"{path.stem}.txt", image.shape[1], image.shape[0])
        if code and box:
            yield path, code, crop(image, box)


def tune(root: Path) -> tuple[dict, list[dict]]:
    rows = []
    samples = list(split_samples(root, "valid"))
    for cfg in configurations():
        errors = []; exact = 0
        for _, code, roi in samples:
            chars, _ = segment(preprocess(roi, cfg)); error = abs(len(chars) - len(code)); errors.append(error); exact += error == 0
        rows.append({"config": cfg, "exact_count_rate": exact/len(samples), "count_mae": float(np.mean(errors))})
    best = max(rows, key=lambda row: (row["exact_count_rate"], -row["count_mae"]))
    return best, rows


def normalize_char(char: np.ndarray, size: int = 32) -> np.ndarray:
    h, w = char.shape; scale = (size - 6) / max(h, w); nw, nh = max(1, int(w*scale)), max(1, int(h*scale))
    resized = cv2.resize(char, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), np.uint8); x, y = (size-nw)//2, (size-nh)//2; canvas[y:y+nh, x:x+nw] = resized
    return canvas


def hog(char: np.ndarray) -> np.ndarray:
    descriptor = cv2.HOGDescriptor((32,32), (16,16), (8,8), (8,8), 9)
    return descriptor.compute(normalize_char(char)).ravel().astype(np.float32)


def train_knn(root: Path, cfg: dict) -> tuple[cv2.ml.KNearest, list[str], dict]:
    labels = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"); lookup = {c:i for i,c in enumerate(labels)}
    features=[]; responses=[]; accepted=total=0
    for _, code, roi in split_samples(root, "train"):
        total += 1; chars, _ = segment(preprocess(roi, cfg))
        if len(chars) != len(code):
            continue
        accepted += 1
        for char, label in zip(chars, code, strict=True):
            features.append(hog(char)); responses.append(lookup[label])
    if not features:
        raise RuntimeError("No exactly segmented training samples")
    model=cv2.ml.KNearest_create(); model.train(np.asarray(features), cv2.ml.ROW_SAMPLE, np.asarray(responses, np.float32))
    return model, labels, {"train_rois_total":total, "train_rois_accepted":accepted, "characters":len(features)}


def evaluate_ocr(root: Path, split: str, cfg: dict, model, labels: list[str], k: int) -> dict:
    chars_ok=chars_total=exact_codes=segmented_codes=total=0
    for _, code, roi in split_samples(root, split):
        total += 1; chars, _ = segment(preprocess(roi, cfg))
        if len(chars) != len(code):
            continue
        segmented_codes += 1; samples=np.asarray([hog(char) for char in chars]); _, result, _, _=model.findNearest(samples, k)
        prediction="".join(labels[int(index)] for index in result.ravel()); exact_codes += prediction == code
        chars_ok += sum(a==b for a,b in zip(prediction, code, strict=True)); chars_total += len(code)
    return {"split":split,"k":k,"images":total,"segmentation_exact_rate":segmented_codes/total if total else 0,"character_accuracy_conditional":chars_ok/chars_total if chars_total else 0,"exact_code_accuracy_end_to_end":exact_codes/total if total else 0,"exact_codes":exact_codes}


def save_stage_figure(path: Path, code: str, stages: dict, boxes: list, output: Path) -> None:
    segmented=cv2.cvtColor(stages["normalized"],cv2.COLOR_BGR2RGB)
    for x,y,w,h in boxes: cv2.rectangle(segmented,(x,y),(x+w,y+h),(0,255,0),2)
    panels=[("YOLO crop",cv2.cvtColor(stages["roi"],cv2.COLOR_BGR2RGB),None),("Rotate + upscale",cv2.cvtColor(stages["normalized"],cv2.COLOR_BGR2RGB),None),("Grayscale",stages["gray"],"gray"),("Contrast",stages["contrast"],"gray"),("Threshold",stages["binary"],"gray"),("Components",segmented,None)]
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for axis,(title,image,cmap) in zip(axes.ravel(),panels,strict=True): axis.imshow(image,cmap=cmap); axis.set_title(title); axis.axis("off")
    fig.suptitle(f"Intermediate pipeline – expected {code}"); fig.savefig(output/f"{path.stem}_stages.png",dpi=180); plt.close(fig)


def detector_comparison(classical_json: Path, output: Path) -> None:
    data=json.loads(classical_json.read_text()); classical=data["validation_winners"]
    names=[x["method"] for x in classical]+["YOLO11n"]; precision=[x["precision"] for x in classical]+[0.996]; recall=[x["recall"] for x in classical]+[0.971]
    f1=[x["f1"] for x in classical]+[2*0.996*0.971/(0.996+0.971)]
    x=np.arange(len(names)); fig,ax=plt.subplots(figsize=(11,6)); width=.25
    ax.bar(x-width,precision,width,label="Precision"); ax.bar(x,recall,width,label="Recall"); ax.bar(x+width,f1,width,label="F1")
    ax.set_xticks(x,names,rotation=15); ax.set_ylim(0,1.08); ax.set_ylabel("Score at IoU 0.5"); ax.set_title("ROI detector comparison (validation-tuned classical methods vs YOLO test)"); ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); fig.savefig(output/"detector_comparison.png",dpi=200); plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--data",default="data"); parser.add_argument("--model",default="outputs/yolo/train/weights/best.pt"); parser.add_argument("--output",default="experiments/hybrid_ocr/results"); args=parser.parse_args()
    root=Path(args.data); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    detector_comparison(Path("experiments/classical_roi/results/comparison.json"),output)
    best, rows=tune(root); (output/"preprocessing_sweep.json").write_text(json.dumps(rows,indent=2)); cfg=best["config"]
    model,labels,training=train_knn(root,cfg); sweeps=[evaluate_ocr(root,"valid",cfg,model,labels,k) for k in (1,3,5,7)]; best_k=max(sweeps,key=lambda x:x["exact_code_accuracy_end_to_end"])["k"]; test=evaluate_ocr(root,"test",cfg,model,labels,best_k)
    summary={"selected_preprocessing":best,"knn_training":training,"knn_validation_sweep":sweeps,"selected_k":best_k,"test":test}; (output/"ocr_metrics.json").write_text(json.dumps(summary,indent=2))
    np.savez_compressed(output/"knn_hog.npz",labels=np.asarray(labels))
    # Intermediate figures use actual YOLO predictions, not ground-truth crops.
    detector=YOLO(args.model)
    for path in sorted((root/"test"/"images").glob("*.jpg"))[:6]:
        image=cv2.imread(str(path)); prediction=detector.predict(image,verbose=False,conf=.25)[0]
        if len(prediction.boxes)==0: continue
        xyxy=prediction.boxes.xyxy[0].cpu().numpy().astype(int); box=(xyxy[0],xyxy[1],xyxy[2]-xyxy[0],xyxy[3]-xyxy[1]); stages=preprocess(crop(image,box),cfg); _,boxes=segment(stages); save_stage_figure(path,code_from_path(path),stages,boxes,output)
    fig,ax=plt.subplots(1,2,figsize=(11,4)); ax[0].bar(range(len(rows)),[x["exact_count_rate"] for x in rows]); ax[0].set_title("Preprocessing parameter sweep"); ax[0].set_xlabel("Configuration index"); ax[0].set_ylabel("Exact character-count rate")
    ax[1].plot([x["k"] for x in sweeps],[x["character_accuracy_conditional"] for x in sweeps],marker="o",label="Character accuracy"); ax[1].plot([x["k"] for x in sweeps],[x["exact_code_accuracy_end_to_end"] for x in sweeps],marker="o",label="Exact code accuracy"); ax[1].set_xlabel("KNN k"); ax[1].set_ylim(0,1); ax[1].legend(); ax[1].set_title("KNN parameter sweep"); fig.tight_layout(); fig.savefig(output/"parameter_sweeps.png",dpi=190); plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
