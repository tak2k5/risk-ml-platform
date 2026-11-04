# Build từ thư mục gốc repo (cần cả riskops/):  docker build -f fraud/Dockerfile -t fraud-consumer .
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY fraud/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY fraud/src ./src
COPY riskops/src ./riskops
ENV PYTHONPATH=/app/src:/app/riskops PYTHONUNBUFFERED=1
CMD ["python", "-m", "fraudrt.consumer"]
