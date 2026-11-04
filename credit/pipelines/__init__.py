"""Dự án #25: giám sát drift và tự động huấn luyện lại cho mô hình rủi ro tín dụng.

drift.py     tính PSI / KS cho từng đặc trưng và cho điểm số mô hình
monitor.py   so sánh dữ liệu mới với dữ liệu train, ghi báo cáo (CLI)
promote.py   quyết định challenger có thay champion hay không
flow.py      luồng Prefect: kiểm tra drift -> retrain -> so sánh -> promote
simulate.py  giả lập 12 tháng có drift để thấy giá trị của việc retrain
"""
