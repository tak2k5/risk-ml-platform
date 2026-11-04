# Build từ thư mục gốc repo (cần cả riskops/):  docker build -f credit/Dockerfile -t credit-risk .

# ---------- Giai đoạn 1: cài thư viện vào một venv riêng ----------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# Mọi thư viện đều có wheel dựng sẵn nên không cần trình biên dịch
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY credit/requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# ---------- Giai đoạn 2: image chạy, nhỏ gọn, không chạy bằng root ----------
FROM python:3.11-slim AS runtime
# LightGBM cần thư viện OpenMP (libgomp) lúc chạy
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src:/app/riskops \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models
WORKDIR /app
COPY credit/src/ src/
COPY riskops/src/ riskops/
# Mô hình được copy vào image (cách đơn giản nhất). Muốn tách rời thì mount volume vào /app/models.
COPY credit/models/ models/
USER 1000
EXPOSE 8000
# Kiểm tra sức khỏe bằng Python có sẵn, không cần cài thêm curl
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=2)" || exit 1
# 1 worker: số đo Prometheus nằm trong bộ nhớ tiến trình; muốn chịu tải hơn thì tăng replica (HPA), không tăng worker
CMD ["uvicorn", "credit_risk.api:app", "--host", "0.0.0.0", "--port", "8000"]
