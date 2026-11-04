# Hướng dẫn cho trợ lý code (AGENTS.md)

Monorepo hai mô hình rủi ro: `credit/` (tín dụng), `fraud/` (gian lận trên luồng), `riskops/` (thư viện dùng chung).
Đọc README.md ở gốc trước, rồi AGENTS.md của thư mục mình sửa.

## Lệnh

- Cài: `make install`. Test cả repo: `make test`. Lint: `make lint`.
- Chạy cả nền tảng: `make train-synthetic up smoke`; tắt: `make down`.
- Trong từng thư mục con, `PYTHONPATH` phải gồm `../riskops/src` (Makefile con đã đặt sẵn).

## Quy tắc

- Đo drift chỉ viết ở `riskops/src/riskops/drift.py`; `credit/pipelines/drift.py` chỉ re-export. Đổi tên số đo trong
  `riskops/prom.py` thì sửa cả `monitoring/grafana/dashboards/risk.json`.
- Ghi và đăng ký MLflow đi qua `riskops.registry`. Tên mô hình `credit-risk`, `fraud-lgbm` và alias
  `champion`/`challenger` là hợp đồng với docker-compose và CI.
- Dockerfile của `credit/` và `fraud/` build từ thư mục gốc (`-f credit/Dockerfile .`) vì cần `riskops/`.
- Sửa chỗ nối giữa các phần thì sửa cả `tests/test_platform.py`.
- Bảng Kết quả chỉ điền số đo trên dữ liệu thật; số giả lập phải ghi rõ là giả lập.
- Không đưa dữ liệu Kaggle, `models/`, `mlflow.db`, API key hay tài liệu nội bộ công ty vào repo.
