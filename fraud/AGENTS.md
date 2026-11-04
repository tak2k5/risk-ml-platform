# Hướng dẫn cho trợ lý code (AGENTS.md)

Chấm điểm gian lận từng giao dịch trên luồng: Redpanda (Kafka) -> consumer (đặc trưng cửa sổ trong Redis + LightGBM)
-> cảnh báo; Prometheus + Grafana theo dõi. Python 3.10+, src layout; đọc README.md trước.

## Lệnh

- Cài: `pip install -r requirements-dev.txt`. Test: `make test` (không cần Docker). Lint: `make lint`.
- Chạy cả luồng: `make train-synthetic up bench-synthetic`; dữ liệu thật: `make data train up bench`.

## Quy tắc

- Mọi đặc trưng đi qua MỘT hàm `features.featurize`, dùng cho cả huấn luyện (phát lại lịch sử) và consumer. Không
  viết lại đặc trưng riêng cho một bên; thêm đặc trưng mới thì `scripts/parity.py` phải vẫn báo không lệch.
- Đặc trưng của một giao dịch chỉ dùng giao dịch TRƯỚC nó (cửa sổ loại chính nó). Chia train/valid/test theo thời gian.
- Không dùng cột `isFlaggedFraud` (quy tắc của hệ cũ, gần như rỗng). Không chọn ngưỡng trên tập test.
- Không đưa `data/`, `models/` vào repo (điều khoản Kaggle). Số trong README chỉ lấy từ `reports/` chạy trên PaySim;
  số trên dữ liệu giả lập phải ghi rõ là giả lập.
