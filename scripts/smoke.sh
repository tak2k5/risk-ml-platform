#!/usr/bin/env bash
# Kiểm tra cả nền tảng đang chạy (make train-synthetic up trước): gửi hồ sơ vay và giao dịch, rồi xem số đo.
set -uo pipefail
PY=${PY:-python}
export PYTHONPATH=credit/src:credit:fraud/src:riskops/src
ok=0; fail=0
check() { if eval "$2" >/dev/null 2>&1; then echo "ĐẠT  $1"; ok=$((ok+1)); else echo "LỖI  $1"; fail=$((fail+1)); fi; }
wait_for() { for _ in $(seq 1 60); do curl -fs "$1" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

wait_for localhost:8000/health; wait_for localhost:8001/metrics; wait_for localhost:5000/health
check "API tín dụng có mô hình" "curl -fs localhost:8000/health | grep -q '\"status\":\"ok\"'"
check "API tín dụng chấm hồ sơ mẫu" "$PY credit/scripts/smoke_test.py http://localhost:8000"
check "Gửi 600 hồ sơ của một tháng lệch phân phối" "$PY scripts/send_credit.py --n 600 --shift 2.0"
check "Phát 3000 giao dịch vào Kafka" "$PY -m fraudrt.producer --synthetic --rate 1000 --limit 3000"
sleep 8
check "Consumer đã chấm giao dịch" "curl -fs localhost:8001/metrics | grep -E '^fraud_events_total [1-9]'"
check "Drift tín dụng lên /metrics (model=credit)" "curl -fs localhost:8000/metrics/ | grep -q 'risk_drift_alert{model=\"credit\"} 1.0'"
check "Drift gian lận lên /metrics (model=fraud)" "curl -fs localhost:8001/metrics | grep -q 'risk_feature_psi{feature=\"amount\",model=\"fraud\"}'"
check "MLflow sống" "curl -fs localhost:5000/health"
sleep 6
check "Prometheus thấy cả hai dịch vụ" \
  "[ \$(curl -fs localhost:9090/api/v1/targets | grep -o '\"health\":\"up\"' | wc -l) -ge 2 ]"
check "Grafana có dashboard chung" "curl -fs -u admin:\${GRAFANA_PASSWORD:-admin} localhost:3000/api/search | grep -q risk-overview"
echo; echo "$ok đạt, $fail lỗi"; [ "$fail" -eq 0 ]
