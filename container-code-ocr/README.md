# Nhận dạng mã container bằng Computer Vision và OCR

Dự án phát hiện vùng chứa mã định danh container trong ảnh, tách ký tự và nhận dạng chuỗi mã bằng xử lý ảnh cổ điển. Source gồm một pipeline chính chạy bằng CLI và các thí nghiệm so sánh detector ROI cổ điển với YOLO.

Pipeline chính không cần GPU hoặc YOLO:

```text
Ảnh đầu vào
  → Grayscale/HSV Value
  → Top-hat + Black-hat → Gaussian blur
  → Adaptive threshold + Canny
  → Morphological closing theo 2 hướng
  → Contour filtering + NMS → Crop ROI
  → Tách ký tự, resize 20 × 30
  → KNN → Chuỗi mã + ảnh kết quả
```

## 1. Chức năng

- Nhận một ảnh, nhiều ảnh hoặc toàn bộ thư mục.
- Lưu năm stage: grayscale, contrast, blur, binary và edge.
- Phát hiện vùng mã ngang/dọc, xếp hạng candidate và loại box trùng bằng NMS.
- Tách ký tự theo contour và nhận dạng bằng OpenCV KNN.
- Tạo tập ký tự từ ảnh và nhãn YOLO ở chế độ tự động hoặc thủ công.
- Sweep tham số; đánh giá ROI bằng Precision, Recall, F1 và mean IoU.
- Có thí nghiệm detector cổ điển và hybrid YOLO + HOG-KNN.

## 2. Cấu trúc project

```text
container-code-ocr/
├── configs/default.yaml            # Tham số pipeline chính
├── data/
│   ├── data.yaml                   # Cấu hình dataset YOLO
│   ├── train/{images,labels}/
│   ├── valid/{images,labels}/
│   ├── test/{images,labels}/
│   └── processed/chars/<class>/    # Class 0-9 và A-Z
├── experiments/
│   ├── classical_roi/              # Scharr, morphology, MSER
│   └── hybrid_ocr/                 # YOLO + segmentation + HOG-KNN
├── outputs/                        # Model, kết quả, metrics, sweep
├── scripts/
├── src/container_ocr/
│   ├── pipeline.py                 # Preprocess, detect, segment, recognize
│   ├── cli.py                      # CLI inference
│   ├── build_char_dataset.py       # Tạo tập ký tự
│   ├── train_knn.py                # Huấn luyện KNN
│   ├── evaluate.py                 # Đánh giá ROI
│   ├── sweep.py                    # Sweep tham số
│   └── viz.py                      # Vẽ và lưu output
└── tests/
```

`data/`, model và phần lớn `outputs/` bị loại khỏi Git. Khi clone sang máy mới, cần tải dataset và tạo/chép model KNN nếu muốn nhận dạng ra text.

## 3. Cài đặt

Yêu cầu Python 3.10 trở lên. Pipeline chính chạy CPU; gán nhãn thủ công cần desktop có thể mở cửa sổ OpenCV.

```bash
cd /home/hongdao/cv/container-code-ocr

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Lệnh cuối cài OpenCV, NumPy, Matplotlib, pandas, PyYAML, tqdm và đăng ký command `container-ocr`.

Để chạy test:

```bash
python -m pip install pytest
```

## 4. Chuẩn bị dataset

Dataset là **Container Character Codes v3 Augmented** trên [Roboflow Universe](https://universe.roboflow.com/public-workspace-n6wxn/container-character-codes), license công bố là CC BY 4.0. Export dạng YOLOv8 hoặc YOLOv5:

```text
data/
├── train/
│   ├── images/*.jpg
│   └── labels/*.txt
├── valid/
│   ├── images/*.jpg
│   └── labels/*.txt
└── test/
    ├── images/*.jpg
    └── labels/*.txt
```

Mỗi dòng nhãn: `class_id center_x center_y width height`, tọa độ chuẩn hóa về `[0, 1]`.

```bash
find data/train/images -type f | wc -l
find data/valid/images -type f | wc -l
find data/test/images -type f | wc -l
```

> Workspace dùng trực tiếp `data/train`, `data/valid`, `data/test`. README cũ và `scripts/run_pipeline.sh` còn trỏ đến `data/raw`; không dùng đường dẫn đó cho demo bên dưới.

## 5. Chạy demo nhanh

### 5.1 Demo một ảnh

Nếu đã có `outputs/knn_chars.npz`:

```bash
python -m container_ocr.cli \
  data/test/images/01012021050110310403_CAIU883333_jpg.rf.5e5fb068bbf88cb97a31a24ad388f7d2.jpg \
  --save-stages \
  --output-dir outputs/demo
```

Hoặc dùng entry point đã cài:

```bash
container-ocr \
  data/test/images/01012021050110310403_CAIU883333_jpg.rf.5e5fb068bbf88cb97a31a24ad388f7d2.jpg \
  --save-stages \
  --output-dir outputs/demo
```

### 5.2 Cả tập test hoặc nhiều đầu vào

```bash
container-ocr data/test/images --save-stages --output-dir outputs/demo

container-ocr image_1.jpg image_2.png path/to/folder --output-dir outputs/demo
```

CLI đọc `.jpg`, `.jpeg`, `.png` viết thường ở ngay trong thư mục, không duyệt đệ quy.

Terminal sẽ in dạng:

```text
data/test/images/example.jpg: CAIU883333
```

Nếu chưa có model KNN hoặc OCR rỗng, CLI có thể in `no-code-found`. Ảnh vẫn được tạo và ROI detect được ghi `code-region`. Hãy xem output để phân biệt “không có ROI” với “có ROI nhưng chưa nhận dạng được”.

### 5.3 Output

```text
outputs/demo/
├── <ten_anh>_result.png
└── intermediate/
    ├── <ten_anh>_gray.png
    ├── <ten_anh>_contrast.png
    ├── <ten_anh>_blur.png
    ├── <ten_anh>_binary.png
    └── <ten_anh>_edges.png
```

- `result`: ảnh gốc, bounding box và text.
- `gray`: kênh Value HSV hoặc grayscale.
- `contrast`: sau Top-hat/Black-hat.
- `blur`: sau Gaussian blur.
- `binary`: adaptive threshold đảo.
- `edges`: biên Canny.

Bỏ `--save-stages` nếu chỉ cần ảnh cuối.

## 6. Pipeline chi tiết

### Bước 1 — Đọc ảnh

`ContainerCodePipeline.process_image()` dùng `cv2.imread`. Ảnh không đọc được gây `FileNotFoundError`.

### Bước 2 — Tiền xử lý

1. Chuyển BGR sang kênh Value HSV theo mặc định, giảm phụ thuộc màu sơn.
2. Top-hat/Black-hat làm nổi ký tự sáng/tối trên nền không đều.
3. Kết hợp với ảnh xám để tăng tương phản.
4. Gaussian blur giảm nhiễu.
5. Adaptive Gaussian threshold tạo foreground trắng.
6. Canny trích xuất biên.

### Bước 3 — Phát hiện ROI

Pipeline OR binary với Canny, rồi closing bằng kernel dài theo phương ngang và dọc. Contour được lọc theo tỷ lệ diện tích, aspect ratio, mật độ foreground và độ chữ nhật. Điểm candidate kết hợp độ chữ nhật, mật độ và log diện tích.

Các box trùng bị loại bằng NMS tại IoU `0.35`; giữ tối đa `max_candidates` vùng.

### Bước 4 — Tách ký tự

Mỗi ROI chạy lại tiền xử lý. Contour được lọc theo diện tích, chiều cao tương đối và aspect ratio. ROI ngang sắp trái sang phải; ROI dọc ưu tiên trên xuống dưới. Ký tự resize về `20 × 30`.

### Bước 5 — KNN

Ký tự được flatten thành vector 600 chiều, chuẩn hóa `[0, 1]`, dự đoán với `k=3` mặc định. Alphabet:

```text
0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ
```

`outputs/knn_chars.npz` lưu `samples`, `responses`, `labels`; model OpenCV được dựng lại khi pipeline khởi tạo.

## 7. Tạo dữ liệu ký tự và train KNN

Cấu trúc yêu cầu:

```text
data/processed/chars/
├── 0/*.png
├── ...
├── A/*.png
└── Z/*.png
```

### 7.1 Tạo tự động

Tên ảnh Roboflow chứa mã thật. Tool crop theo nhãn YOLO và chỉ tự gán nhãn khi số ký tự tách được bằng độ dài mã:

```bash
python -m container_ocr.build_char_dataset \
  --data data --split train \
  --output data/processed/chars --mode auto
```

Thử tối đa 100 ký tự:

```bash
python -m container_ocr.build_char_dataset --mode auto --limit 100
```

Metadata được nối vào `metadata.jsonl`; `sample_id` đã hoàn tất sẽ được bỏ qua khi chạy lại.

### 7.2 Gán nhãn thủ công

```bash
python -m container_ocr.build_char_dataset \
  --data data --split train \
  --output data/processed/chars --mode manual
```

- `0-9` / `A-Z`: gán class.
- `Enter`: nhận gợi ý.
- `S`: bỏ qua.
- `Q` / `Esc`: thoát.

Chế độ này dùng `cv2.imshow`, không phù hợp server headless.

### 7.3 Train

```bash
python -m container_ocr.train_knn \
  --data-dir data/processed/chars \
  --output outputs/knn_chars.npz
```

Nếu báo `No character images found`, kiểm tra tên folder class và ảnh bên trong.

## 8. Cấu hình

Mặc định đọc `configs/default.yaml`:

```bash
container-ocr data/test/images/example.jpg \
  --config configs/my_config.yaml --save-stages
```

| Nhóm | Tham số | Ý nghĩa |
|---|---|---|
| `paths` | `model`, `output_dir`, `char_train_dir` | Model, output, dữ liệu ký tự |
| `preprocess` | `morph_kernel` | Kernel Top-hat/Black-hat |
| `preprocess` | `gaussian_kernel` | Kernel Gaussian; số chẵn tự tăng thành lẻ |
| `preprocess` | `adaptive_block`, `adaptive_c` | Adaptive threshold |
| `preprocess` | `canny_low`, `canny_high` | Ngưỡng Canny |
| `detection` | `close_kernel_long/short` | Nối nét theo hai hướng |
| `detection` | `min/max_area_ratio`, `min/max_aspect` | Lọc ROI |
| `detection` | `padding`, `max_candidates` | Padding, số ROI |
| `segmentation` | `min_char_*`, `max_char_*` | Lọc contour ký tự |
| `segmentation` | `char_width`, `char_height` | Kích thước feature |
| `recognition` | `k`, `alphabet` | KNN và class |

Đổi kích thước ký tự thì phải train lại model.

## 9. Sweep tham số

```bash
python -m container_ocr.sweep \
  data/test/images/01012021050110310403_CAIU883333_jpg.rf.5e5fb068bbf88cb97a31a24ad388f7d2.jpg \
  --output-dir outputs/sweeps
```

Tool tạo panel cho:

- `gaussian_kernel`: 3, 5, 9.
- `adaptive_c`: 3, 9, 15.
- `canny_low`: 30, 60, 100.
- `morph_kernel`: 9, 17, 31.

## 10. Đánh giá ROI

```bash
python -m container_ocr.evaluate \
  --images data/test/images \
  --labels data/test/labels \
  --iou-threshold 0.5 \
  --output outputs/eval/metrics.json
```

- `metrics.json`: tổng hợp.
- `metrics.csv`: `gt`, `pred`, `tp`, `fp`, `fn`, `mean_iou` theo ảnh.

Matching one-to-one; box là TP nếu ground truth tốt nhất chưa dùng có IoU ≥ ngưỡng.

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1        = 2 × Precision × Recall / (Precision + Recall)
```

## 11. Thí nghiệm mở rộng

### 11.1 Detector cổ điển

So sánh Scharr + Black-hat, morphology và MSER. Tune trên `valid`, chỉ đánh giá phương pháp thắng một lần trên `test`:

```bash
python experiments/classical_roi/run_experiment.py \
  --data data \
  --output experiments/classical_roi/results
```

Kết quả gồm `comparison.json`, `validation_f1.png` và ảnh ví dụ.

### 11.2 Hybrid YOLO + HOG-KNN

Dependency bổ sung, không nằm trong `pyproject.toml`:

```bash
python -m pip install ultralytics
```

Cần trọng số `outputs/yolo/train/weights/best.pt`, file `experiments/classical_roi/results/comparison.json` và đủ ba split:

```bash
python experiments/hybrid_ocr/run_pipeline.py \
  --data data \
  --model outputs/yolo/train/weights/best.pt \
  --output experiments/hybrid_ocr/results
```

Hybrid thử CLAHE/Otsu/adaptive threshold/morphology, connected components, HOG 32 × 32, KNN, weighted voting và ràng buộc chữ/số.

**Giới hạn:** implementation hiện tại tune/train/đánh giá OCR trên ROI từ **ground-truth YOLO labels**. Model YOLO qua `--model` dùng sinh hình intermediate trên một số ảnh test. Do đó `ocr_metrics.json` đo OCR khi ROI đã đúng, chưa phải accuracy end-to-end YOLO nối trực tiếp OCR.

## 12. Chạy test

```bash
python -m pytest tests -q
```

Test kiểm tra đủ năm stage, đọc mã từ tên Roboflow, chuẩn hóa ký tự 20 × 30 và lưu đúng class.

## 13. Kịch bản demo đề xuất

1. Mở ảnh gốc, chỉ ra mã cần nhận dạng.
2. Chạy mục 5.1 với `--save-stages`.
3. Mở `gray → contrast → binary → edges`.
4. Mở `*_result.png` xem box và text.
5. Chạy sweep để so sánh tham số.
6. Chạy evaluation, trình bày Precision, Recall, F1, mean IoU.
7. Nếu cần, trình bày thí nghiệm mục 11.

## 14. Lỗi thường gặp

### Không import được `container_ocr`

```bash
cd /home/hongdao/cv/container-code-ocr
source .venv/bin/activate
python -m pip install -e .
```

### CLI không xử lý ảnh

Kiểm tra đường dẫn/đuôi ảnh. CLI chỉ nhận `.jpg`, `.jpeg`, `.png` viết thường và không quét thư mục con.

### Có box nhưng không có text

```bash
ls -lh outputs/knn_chars.npz
```

Nếu chưa có model, tạo dữ liệu và train theo mục 7.

### `Cannot read image` / `FileNotFoundError`

Chạy từ thư mục gốc `container-code-ocr`, vì đường dẫn mặc định là tương đối.

### Lỗi `qt.qpa...` khi gán nhãn

Môi trường headless. Dùng `--mode auto` hoặc chạy manual trên máy có desktop.

### Thiếu `ultralytics`

Chỉ hybrid cần package này. Cài theo mục 11.2 hoặc dùng pipeline chính.

## 15. Liên hệ nội dung môn học

- Xử lý ảnh: không gian màu, morphology, Gaussian filtering, adaptive threshold.
- Phát hiện: Canny, contour, connected components, region proposal.
- Phân đoạn: lọc contour ký tự bằng đặc trưng hình học.
- Nhận dạng: KNN trên pixel; thí nghiệm mở rộng dùng HOG-KNN.

ROI được đo bằng Precision/Recall/F1 tại IoU 0.5 và mean IoU. Hybrid phân tích thêm character accuracy, character error rate và exact-code accuracy.
