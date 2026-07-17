# NHẬT KÝ THỰC NGHIỆM: BIẾN ĐỔI AFFINE VÀ KNN

- Thời điểm UTC: 2026-07-17T14:30:05.929216+00:00
- Lệnh đã chạy: `python3 experiments/affine_recognition/run_experiment.py`
- Python: 3.12.3; OpenCV: 4.13.0; NumPy: 2.3.3
- Seed: 42; dữ liệu: `data/processed/chars`
- Output: `experiments/affine_recognition/results`

## 1. Mục tiêu và lưu ý khái niệm

Affine là phép biến đổi hình học, không tự sinh ra nhãn ký tự; KNN là bộ phân lớp.
Vì vậy thực nghiệm tách hai giả thuyết: (1) thêm affine trước KNN có tốt hơn không,
và (2) nếu bỏ KNN thì affine + nearest class prototype có đủ tốt không.

## 2. Những gì đã implement

1. Otsu và tự đảo polarity để foreground luôn trắng.
2. Crop bounding box foreground, scale giữ tỷ lệ và đặt giữa canvas 20×30.
3. Ước lượng độ nghiêng `s = mu11 / mu02` từ moment bậc hai.
4. Warp affine theo `x' = x - s(y-cy)`, rồi tịnh tiến centroid về tâm.
5. So sánh KNN pixel gốc, affine + KNN, affine + một prototype trung bình mỗi lớp.
6. Chia train/valid/test theo group ảnh gốc; chọn k trên valid, test chỉ dùng một lần.

## 3. Dữ liệu và thiết lập

- Tổng 1203 ảnh, 33 lớp hiện diện.
- Train/valid/test: 827/147/229 ảnh.
- Số group train/valid/test: {'train': 210, 'valid': 38, 'test': 56}.
- K được chọn: pixel KNN=1, affine KNN=1.
- Lớp không xuất hiện ở test: không có.

## 4. Kết quả test

| Phương pháp | Accuracy | 95% CI | Macro recall | ms/ký tự | Model KiB |
|---|---:|---:|---:|---:|---:|
| KNN gốc (pixel) | 0.6507 | [0.5869, 0.7095] | 0.5666 | 0.0098 | 1941.51 |
| Affine + KNN | 0.4803 | [0.4165, 0.5448] | 0.4255 | 0.0098 | 1941.51 |
| Affine + prototype (không KNN) | 0.3144 | [0.2578, 0.3772] | 0.3328 | 0.0308 | 77.47 |

## 5. Kết luận

Phương pháp tốt nhất trên test là **KNN gốc (pixel)**. Affine + KNN đổi -0.1703 accuracy so với KNN gốc; nhánh bỏ KNN đổi -0.3362. Trong split này, chưa có bằng chứng rằng affine-prototype cải thiện KNN; nên xem affine là bước chuẩn hoá, không phải phép thay thế classifier.

Kết quả chỉ đo **nhận dạng ký tự đã được crop** trong `data/processed/chars`,
không phải độ chính xác end-to-end của detect ROI → segment → OCR.

## 6. Nhật ký triển khai và kiểm chứng

1. Đọc pipeline, xác nhận KNN là classifier ký tự còn affine là biến đổi hình học.
2. Kiểm kê 1.203 ảnh/33 lớp; nhận thấy các biến thể cùng nguồn cần group split.
3. Thêm module affine, CLI train prototype, config riêng, tích hợp pipeline và test.
4. `py_compile` pass; 3 test affine/pipeline chạy trực tiếp đều pass.
5. Không chạy được pytest suite vì Python hệ thống chưa cài package `pytest`.
6. Lần experiment đầu gặp khác biệt API OpenCV Python (không có
   `KNearest.getTrainSamples`); đã bỏ call thừa và chạy lại thành công.
7. Audit split đầu thấy test thiếu 8 lớp; đã thêm kiểm soát class coverage và
   chạy lại. Bản cuối có đủ 33 lớp trong test và không trộn group giữa split.
8. Train model full-data thành công: 33 prototype từ 1.203 ảnh.
9. Smoke test CLI affine chạy xong trên ảnh CAIU883333; pipeline end-to-end dự đoán
   `1`, cho thấy detect/segment hiện vẫn là nút thắt và không được tính vào metric ký tự.

## 7. File kết quả

- `metrics.json`: toàn bộ cấu hình, metric và CI.
- `predictions.csv`: dự đoán từng ảnh test.
- `per_class_metrics.csv`: recall theo lớp.
- `splits.csv`: split tái lập và group chống leakage.
- `affine_test_model.npz`: prototype train-only dùng trong phép đo test.
- `affine_steps.png`: ảnh trung gian từng bước.
- `metrics_comparison.png`: accuracy, latency, kích thước model.
- `confusion_matrices.png`: confusion matrix ba phương pháp.
- `class_prototypes.png`: prototype trung bình của từng lớp.
- `prediction_changes.png`: ví dụ dự đoán thay đổi.

## 8. Cách chạy lại

```bash
cd /home/hongdao/cv/container-code-ocr
PYTHONPATH=src python3 experiments/affine_recognition/run_experiment.py

# Train model affine trên toàn bộ dữ liệu để dùng pipeline chính
PYTHONPATH=src python3 -m container_ocr.train_affine --config configs/affine.yaml

# Chạy OCR với affine-prototype và lưu các stage pipeline ảnh
PYTHONPATH=src python3 -m container_ocr.cli data/test/images --config configs/affine.yaml --save-stages
```

Lưu ý: model `outputs/affine_prototypes.npz` train trên toàn bộ dữ liệu phục vụ demo;
metric báo cáo dùng model train-only riêng để không nhìn test.
