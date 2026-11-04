PY ?= python
SUBS := riskops credit fraud
ALLPATH := credit/src:credit:fraud/src:riskops/src

.PHONY: help install test test-integration lint train-synthetic register up smoke produce down

help:              ## Liệt kê lệnh
	@grep -E '^[a-z-]+:.*##' Makefile | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'
install:           ## Cài thư viện để chạy test cả 3 phần (CPU)
	$(PY) -m pip install -r credit/requirements-dev.txt -r fraud/requirements-dev.txt -r riskops/requirements.txt
test:              ## Test từng phần rồi test xuyên suốt (dữ liệu giả lập, khoảng 1 phút)
	@for d in $(SUBS); do echo "== $$d"; (cd $$d && NO_PREFECT=1 PYTHONPATH=src:../riskops/src $(PY) -m pytest -q) || exit 1; done
	$(MAKE) test-integration
test-integration:  ## Hai mô hình, một MLflow, nạp champion từ registry, drift của cả hai trên /metrics
	NO_PREFECT=1 PYTHONPATH=$(ALLPATH) $(PY) -m pytest -q tests
lint:              ## Kiểm tra code style toàn repo
	ruff check . && ruff format --check .
train-synthetic:   ## Train nhanh cả hai trên dữ liệu GIẢ LẬP -> credit/models, fraud/models (không cần MLflow)
	cd credit && REPORTS_DIR=reports/synthetic PYTHONPATH=src:../riskops/src $(PY) -m credit_risk.train --synthetic 20000 --no-mlflow
	cd fraud && PYTHONPATH=src:../riskops/src $(PY) -m fraudrt.train --synthetic --reports reports/synthetic
register:          ## Train cả hai, ghi vào MLflow đang chạy (make up), duyệt champion. Thêm ARGS=--synthetic để chạy thử
	PYTHONPATH=$(ALLPATH) MLFLOW_TRACKING_URI=http://localhost:5000 $(PY) scripts/register.py $(ARGS)
up:                ## Bật cả nền tảng (cần credit/models và fraud/models)
	docker compose up -d --build
smoke:             ## Gửi hồ sơ vay + giao dịch vào nền tảng đang chạy, kiểm tra số đo và dashboard
	PY=$(PY) bash scripts/smoke.sh
produce:           ## Phát giao dịch 200/giây: PaySim nếu đã có fraud/data/paysim.csv, không thì giả lập
	@if [ -f fraud/data/paysim.csv ]; then src="--data fraud/data/paysim.csv"; else src="--synthetic"; fi; \
	  echo "nguồn: $$src"; PYTHONPATH=$(ALLPATH) $(PY) -m fraudrt.producer $$src --rate 200
down:              ## Tắt và xóa dữ liệu
	docker compose down -v
