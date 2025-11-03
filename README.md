## 1) Tạo môi trường ảo, kích hoạt và cài dependencies

1. Kiểm tra phiên bản Python (khuyến nghị >= 3.10):

```cmd
python --version
```

2. Tạo virtual environment trong thư mục dự án:

```cmd
python -m venv .venv
```

3. Kích hoạt environment trên Windows (cmd.exe):

```cmd
.venv\Scripts\activate
```

4. Cập nhật pip và cài các gói từ `requirements.txt`:

```cmd
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Kiểm tra nhanh (ví dụ PyTorch nếu dự án cần):

```cmd
python -c "import torch; print(torch.__version__)"
```

Nếu có lỗi thiếu package, cài thêm theo thông báo lỗi hoặc liên hệ để được hỗ trợ.

## 2) Chạy `realtime.py` để test mô hình realtime
Mô hình đang thí nghiệm trên 5 ký hiệu: A, B, C, D và "Xin chào"

Chuẩn bị trước khi chạy:
- Đảm bảo file model `sign_sstcn_attention_model.pth` nằm trong thư mục dự án.
- Webcam hoặc camera thiết bị phải hoạt động.
- Đang ở trong virtualenv (đã kích hoạt `.venv`).

Chạy script realtime:

```cmd
python realtime.py
```

Ghi chú:
- Nếu `realtime.py` nhận tham số (ví dụ `--model`, `--device`), bạn có thể chạy kèm tham số. Ví dụ:

```cmd
python realtime.py --model sign_sstcn_attention_model.pth --device cpu
```

## 3) Xem chi tiết mô hình — mở `vsl-sstcn-attention.ipynb`

## 4) Thư mục `data` chứa các chuỗi skeleton data được trích xuất từ video với mediapipe
