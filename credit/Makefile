# Các lệnh hay dùng. Gõ "make help" để xem danh sách.
PY ?= python
export PYTHONPATH := src:../riskops/src

.PHONY: help install data train train-synthetic test lint api docker monitor simulate

help:            ## Liệt kê lệnh
	@grep -E '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install:         ## Cài thư viện cho phát triển
	$(PY) -m pip install -r requirements-dev.txt

data:            ## Tải dữ liệu Kaggle (cần kaggle.json)
	bash scripts/download_data.sh

train:           ## Huấn luyện trên dữ liệu thật
	$(PY) -m credit_risk.train --tune 30

train-synthetic: ## Huấn luyện nhanh trên 20k hồ sơ giả lập
	REPORTS_DIR=reports/synthetic $(PY) -m credit_risk.train --synthetic 20000 --no-mlflow

test:            ## Chạy toàn bộ test
	$(PY) -m pytest

lint:            ## Kiểm tra code style
	ruff check . && ruff format --check .

api:             ## Chạy API ở http://localhost:8000/docs
	uvicorn credit_risk.api:app --reload

docker:          ## Build và chạy API + MLflow bằng Docker Compose
	docker compose up --build

monitor:         ## (#25) So sánh dữ liệu mới với dữ liệu train, ghi báo cáo drift
	$(PY) -m pipelines.monitor --synthetic-shift 0.5

simulate:        ## (#25) Giả lập 12 tháng drift + retrain tự động
	$(PY) -m pipelines.simulate --months 12
