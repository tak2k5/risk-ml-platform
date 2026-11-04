# Hướng dẫn cho trợ lý code (AGENTS.md)

Repo chấm điểm rủi ro tín dụng (Python 3.10+, src layout). Đọc README.md trước.

## Lệnh

- Cài: `pip install -r requirements-dev.txt`
- Test: `PYTHONPATH=src pytest` (24 test, dữ liệu giả lập, khoảng 10 giây)
- Lint: `ruff check . && ruff format --check .`
- Train nhanh: `PYTHONPATH=src python -m credit_risk.train --synthetic 20000 --no-mlflow`

## Quy tắc

- Mọi biến đổi đặc trưng phải nằm trong `FeatureBuilder` (src/credit_risk/features.py), không xử lý riêng ở API.
- Không dùng tập test để chọn mô hình, tham số hay ngưỡng; chỉ dùng validation.
- Không đưa dữ liệu Kaggle, `models/`, `mlruns/`, `mlflow.db` lên git.
- Không thêm `CODE_GENDER` hay thuộc tính nhạy cảm khác vào đặc trưng.
- Số liệu trong README phải lấy từ dữ liệu thật; dữ liệu giả lập chỉ để test.
- Thêm tính năng thì thêm test trong `tests/`.
