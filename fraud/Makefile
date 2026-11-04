PY ?= python
export PYTHONPATH := src:../riskops/src

.PHONY: help install test lint data train train-synthetic up down produce bench bench-synthetic

help:              ## Liệt kê lệnh
	@grep -E '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'
install:           ## Cài thư viện (máy local, Python 3.10+)
	$(PY) -m pip install -r requirements-dev.txt
test:              ## Test offline, không cần Docker (khoảng 20 giây)
	$(PY) -m pytest
lint:              ## Kiểm tra code style
	ruff check . && ruff format --check .
data:              ## Tải PaySim từ Kaggle vào data/paysim.csv (cần kaggle.json)
	bash scripts/download_data.sh
train:             ## Huấn luyện LightGBM trên PaySim -> models/, reports/train.md
	$(PY) -m fraudrt.train --data data/paysim.csv
train-synthetic:   ## Huấn luyện trên dữ liệu giả lập (vài giây, KHÔNG phải kết quả)
	$(PY) -m fraudrt.train --synthetic --reports reports/synthetic
up:                ## Bật Redpanda, Redis, consumer, Prometheus, Grafana (cần models/)
	docker compose up -d --build
down:              ## Tắt và xóa dữ liệu của stack
	docker compose down -v
produce:           ## Phát PaySim phần test vào topic, 200 giao dịch/giây (xem Grafana localhost:3000)
	$(PY) -m fraudrt.producer --data data/paysim.csv --rate 200
bench:             ## Đo độ trễ + kiểm tra lệch đặc trưng trên PaySim -> reports/stream.md
	PY=$(PY) bash scripts/bench.sh
bench-synthetic:   ## Như bench nhưng với dữ liệu giả lập (cần train-synthetic trước)
	PY=$(PY) SYNTH=1 bash scripts/bench.sh
