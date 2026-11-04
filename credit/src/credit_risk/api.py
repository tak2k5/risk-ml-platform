"""API chấm điểm tín dụng bằng FastAPI.

Chạy local:
    uvicorn credit_risk.api:app --reload
Mở http://localhost:8000/docs để thử trực tiếp trên trình duyệt.

Nạp mô hình từ thư mục MODEL_DIR (mặc định), hoặc từ MLflow registry dùng chung nếu đặt MODEL_URI, ví dụ
MODEL_URI=models:/credit-risk@champion (cần MLFLOW_TRACKING_URI). /metrics cho Prometheus gồm số hồ sơ theo quyết
định, độ trễ, phân phối điểm, và PSI trên cửa sổ các hồ sơ gần nhất so với dữ liệu train (riskops.StreamDrift).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app

from .config import CATEGORICAL_COLS, NUMERIC_COLS, RAW_COLS, get_settings
from .explain import Explainer
from .schemas import Application, BatchRequest, BatchResponse, Prediction, ReasonOut

log = logging.getLogger("credit_risk.api")

# Số đo cho Prometheus. Chạy 1 worker mỗi container (Kubernetes nhân bản bằng replica), nên mỗi tiến trình một bộ đếm.
DECISIONS = Counter("credit_decisions_total", "Số hồ sơ đã chấm theo quyết định", ["decision"])
LATENCY = Histogram("credit_predict_seconds", "Thời gian chấm điểm (gồm SHAP)", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))  # fmt: skip
SCORES = Histogram("credit_probability_default", "Phân phối xác suất vỡ nợ",
                   buckets=(0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1))  # fmt: skip


class ModelService:
    """Giữ mô hình, metadata và explainer trong bộ nhớ; nạp một lần khi khởi động."""

    def __init__(self, model_path: Path, metadata_path: Path):
        if not model_path.exists():
            raise FileNotFoundError(f"Chưa có mô hình tại {model_path}. Hãy chạy: python -m credit_risk.train")
        self.pipeline = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        self.threshold = float(self.metadata.get("threshold", 0.5))
        self.version = str(self.metadata.get("model_version", "unknown"))
        self.explainer = Explainer(self.pipeline)
        self.drift = None
        ref_path = model_path.parent / "reference.csv.gz"
        if ref_path.exists():
            self.drift = self._drift_tracker(pd.read_csv(ref_path)[RAW_COLS])

    def _drift_tracker(self, ref: pd.DataFrame):
        """PSI trên DRIFT_WINDOW hồ sơ gần nhất, tính lại mỗi DRIFT_EVERY hồ sơ, đẩy lên /metrics."""
        from riskops.drift import StreamDrift
        from riskops.prom import DriftGauges

        window = int(os.getenv("DRIFT_WINDOW", "1000"))
        self.gauges = DriftGauges("credit")
        return StreamDrift(
            ref,
            window=window,
            every=int(os.getenv("DRIFT_EVERY", "200")),
            min_events=int(os.getenv("DRIFT_MIN_EVENTS", str(min(window, 500)))),
            ref_scores=self.pipeline.predict_proba(ref)[:, 1],
            numeric=NUMERIC_COLS,
            categorical=CATEGORICAL_COLS,
        )

    def predict(self, apps: list[Application]) -> list[Prediction]:
        df = pd.DataFrame([a.model_dump() for a in apps])[RAW_COLS]
        probs = self.pipeline.predict_proba(df)[:, 1]
        reasons = self.explainer.top_reasons(df, k=3)
        for row, p in zip(df.to_dict("records"), probs):
            DECISIONS.labels("reject" if p >= self.threshold else "approve").inc()
            SCORES.observe(float(p))
            if self.drift is not None and (summary := self.drift.update(row, float(p))) is not None:
                self.gauges.publish(summary, len(self.drift.rows))
        return [
            Prediction(
                probability_default=round(float(p), 4),
                decision="reject" if p >= self.threshold else "approve",
                threshold=self.threshold,
                reasons=[ReasonOut(**r.to_dict()) for r in rs],
                model_version=self.version,
            )
            for p, rs in zip(probs, reasons)
        ]


def _model_dir() -> Path:
    """Thư mục chứa model.joblib: MODEL_DIR, hoặc tải từ MLflow nếu đặt MODEL_URI."""
    s = get_settings()
    uri = os.getenv("MODEL_URI")
    if not uri:
        return s.model_dir
    from riskops.registry import fetch

    return fetch(s.mlflow_uri, uri, Path(tempfile.mkdtemp(prefix="credit-model-")))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_dir = _model_dir()
        app.state.service = ModelService(model_dir / "model.joblib", model_dir / "metadata.json")
        log.info("Đã nạp mô hình phiên bản %s", app.state.service.version)
    except FileNotFoundError as exc:
        # Vẫn cho API khởi động để /health báo rõ lỗi, thay vì container chết không rõ lý do
        log.error(str(exc))
        app.state.service = None
    yield


app = FastAPI(
    title="Credit Risk API",
    description="Chấm điểm rủi ro vỡ nợ và giải thích bằng SHAP",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Ghi thời gian xử lý vào header để đo độ trễ khi load test."""
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.1f}"
    return response


def _service(request: Request) -> ModelService:
    service = request.app.state.service
    if service is None:
        raise HTTPException(status_code=503, detail="Mô hình chưa được nạp")
    return service


@app.get("/health")
def health(request: Request) -> dict:
    """Kubernetes/Docker gọi endpoint này để biết dịch vụ còn sống và sẵn sàng chưa."""
    service = request.app.state.service
    return {"status": "ok" if service else "no_model", "model_version": service.version if service else None}


@app.get("/model-info")
def model_info(request: Request) -> dict:
    s = _service(request)
    keys = ["model_version", "trained_at", "data_source", "threshold", "test_metrics"]
    return {k: s.metadata.get(k) for k in keys}


@app.post("/predict", response_model=Prediction)
def predict(application: Application, request: Request) -> Prediction:
    with LATENCY.labels("predict").time():
        return _service(request).predict([application])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest, request: Request) -> BatchResponse:
    with LATENCY.labels("batch").time():
        return BatchResponse(predictions=_service(request).predict(body.applications))
