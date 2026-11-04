# Dữ liệu

Dự án dùng file `application_train.csv` của cuộc thi Kaggle
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
(khoảng 307 nghìn hồ sơ vay, 122 cột, cột mục tiêu `TARGET` = 1 nếu khách trễ hạn).

Theo điều khoản cuộc thi, dữ liệu **không được phát tán lại**, nên repo chỉ chứa script tải.

## Cách tải

1. Đăng nhập Kaggle, mở trang cuộc thi, bấm **Join competition** và đồng ý điều khoản.
2. Vào *Settings → API → Create New Token* để tải `kaggle.json`, đặt vào `~/.kaggle/kaggle.json`
   (trên Colab: tải file lên rồi chạy ô cài đặt trong notebook).
3. Chạy:

```bash
bash scripts/download_data.sh
```

## Không có dữ liệu thật?

Mọi lệnh đều chạy được với dữ liệu giả lập (cùng tên cột, phân phối gần giống, tỉ lệ vỡ nợ ~8%):

```bash
python -m credit_risk.train --synthetic 20000
```

Dữ liệu giả lập chỉ dùng để kiểm thử code và CI. **Số liệu ghi vào README/CV phải lấy từ dữ liệu thật.**

## Cột được dùng

Chỉ đọc 20 cột (xem `src/credit_risk/config.py`). Cột `CODE_GENDER` bị loại có chủ đích để mô hình
không quyết định cho vay dựa trên giới tính.
