# Huấn luyện (dữ liệu GIẢ LẬP, 2026-11-02 00:06)

48,965 giao dịch, gian lận 1.385%. Chia theo thời gian: train 34,275, valid 7,345, test 7,345. PR-AUC của đoán ngẫu nhiên = tỉ lệ gian lận trong test = 0.0131.
Tính đặc trưng (phát lại toàn bộ lịch sử): 2 giây.

| mô hình                     |   threshold |   PR-AUC |   ROC-AUC |   precision |   recall |   cảnh báo |
|:----------------------------|------------:|---------:|----------:|------------:|---------:|-----------:|
| LightGBM + đặc trưng cửa sổ |      0.3953 |   0.924  |    0.9975 |      0.8925 |   0.8646 |         93 |
| LightGBM, không cửa sổ      |      0.7389 |   0.8798 |    0.9962 |      0.9571 |   0.6979 |         70 |

Ngưỡng chọn trên valid (F1 lớn nhất); precision / recall / số cảnh báo đo trên test.

Đặc trưng quan trọng nhất (gain): dest_balance_error (19%), orig_emptied (13%), type_TRANSFER (12%), amount (9%), amount_over_balance (9%), type_CASH_OUT (8%)
