#!/usr/bin/env bash
# Đo độ trễ trên luồng + kiểm tra lệch đặc trưng, trên stack docker compose đang chạy (make up).
#   N=5000 RATE=300 bash scripts/bench.sh            # PaySim (data/paysim.csv)
#   N=5000 RATE=300 SYNTH=1 bash scripts/bench.sh    # dữ liệu giả lập
# Các bước: dừng consumer chạy liên tục -> xóa Redis và tạo lại topic (để không đọc lại tin cũ) -> bật một
# consumer đo đạc (--max-events N, ghi đặc trưng online) -> producer phát N giao dịch -> so với đặc trưng offline.
set -euo pipefail
N=${N:-5000}; RATE=${RATE:-300}; START=${START:-0.85}; PY=${PY:-python}
DATA_ARGS=$([ "${SYNTH:-0}" = 1 ] && echo "--synthetic" || echo "--data data/paysim.csv")
RPK="docker compose exec -T redpanda rpk -X brokers=redpanda:9092"

docker compose stop consumer
docker compose exec -T redis redis-cli flushall >/dev/null
$RPK topic delete transactions >/dev/null 2>&1 || true
$RPK topic create transactions -p 1 >/dev/null
mkdir -p reports
docker rm -f fraud-bench >/dev/null 2>&1 || true
docker compose run -d --no-deps --name fraud-bench consumer python -m fraudrt.consumer \
  --brokers redpanda:9092 --redis redis://redis:6379/0 --group "bench-$(date +%s)" --metrics-port 0 \
  --max-events "$N" --report reports/stream.md --log-features reports/online_features.jsonl >/dev/null
until docker logs fraud-bench 2>&1 | grep -q "sẵn sàng"; do sleep 1; done   # chờ consumer nạp xong mô hình

PYTHONPATH=src:../riskops/src $PY -m fraudrt.producer $DATA_ARGS --rate "$RATE" --start "$START" --limit "$N"
docker wait fraud-bench >/dev/null && docker rm fraud-bench >/dev/null
cat reports/stream.md
PYTHONPATH=src:../riskops/src $PY scripts/parity.py --online reports/online_features.jsonl $DATA_ARGS --start "$START"
docker compose start consumer >/dev/null
