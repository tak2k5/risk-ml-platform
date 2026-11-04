"""Consumer: đọc topic "transactions", chấm điểm từng giao dịch, cảnh báo, xuất số liệu cho Prometheus.

    python -m fraudrt.consumer --max-events 5000 --report reports/stream.md
- Độ trễ mỗi sự kiện = lúc chấm xong - thời điểm producer gửi (timestamp của message Kafka).
- Commit offset SAU khi xử lý: consumer chết giữa chừng thì xử lý lại vài giao dịch, đặc trưng không sai vì Redis
  không đếm trùng (ZADD cùng tx_id).
- --log-features ghi đặc trưng online ra file để scripts/parity.py so với đặc trưng offline.
- Drift trên luồng: PSI từng đặc trưng và của điểm số trên --drift-window giao dịch gần nhất so với mẫu train
  (models/reference.parquet), tính lại mỗi --drift-every giao dịch, xuất `risk_feature_psi{model="fraud"}` ...
  (riskops, dùng chung với API tín dụng).
- MODEL_URI=models:/fraud-lgbm@champion: nạp mô hình từ MLflow dùng chung thay vì thư mục --models.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .alerts import from_env
from .features import RedisWindows
from .scorer import Scorer

log = logging.getLogger("fraudrt.consumer")


class Metrics:
    def __init__(self, port: int | None = 8001):
        from prometheus_client import Counter, Histogram, start_http_server

        buckets = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5)
        self.latency = Histogram("fraud_event_latency_seconds", "Từ lúc gửi tới lúc chấm xong", buckets=buckets)
        self.score_time = Histogram("fraud_score_seconds", "Thời gian đặc trưng + mô hình", buckets=buckets)
        self.events = Counter("fraud_events_total", "Số giao dịch đã chấm")
        self.alerts = Counter("fraud_alerts_total", "Số cảnh báo")
        self.labelled = Counter("fraud_labelled_total", "Giao dịch có nhãn gian lận (để so, chỉ có khi phát lại)")
        if port:
            start_http_server(port)


# Giờ trong ngày đổi theo chu kỳ: cửa sổ vài giờ gần nhất luôn "lệch" so với mẫu cả ngày, nên không đưa vào PSI.
DRIFT_EXCLUDE = ["hour_of_day"]


class DriftMonitor:
    """Nối StreamDrift (tính PSI) với gauge Prometheus. Không có reference.parquet thì không làm gì.

    Lưu ý: đặc trưng cửa sổ (cnt_1h, cnt_24h...) chỉ đúng khi Redis đã có đủ 24 giờ lịch sử; ngay sau khi xóa Redis
    chúng thấp hơn lúc train và PSI sẽ báo lệch, đó là cảnh báo đúng (consumer đang "lạnh").
    """

    def __init__(self, model_dir: Path, window: int, every: int, publish: bool = True):
        from riskops.drift import StreamDrift

        self.tracker, self.gauges = None, None
        path = Path(model_dir) / "reference.parquet"
        if not path.exists() or window <= 0:
            log.warning("không đo drift: thiếu %s (train lại để có)", path)
            return
        ref = pd.read_parquet(path)
        self.tracker = StreamDrift(ref.drop(columns=["score", *DRIFT_EXCLUDE]), window=window, every=every,
                                   ref_scores=ref["score"].to_numpy())  # fmt: skip
        if publish:
            from riskops.prom import DriftGauges

            self.gauges = DriftGauges("fraud")

    def update(self, features: dict, score: float):
        if self.tracker is None:
            return None
        summary = self.tracker.update(features, score)
        if summary is not None:
            if self.gauges:
                self.gauges.publish(summary, len(self.tracker.rows))
            if summary.drifted:
                top = summary.table.head(3)
                log.warning("DRIFT trên luồng: %s; PSI điểm số %s", ", ".join(f"{f}={p:.2f}" for f, p in
                            zip(top.feature, top.psi, strict=True)), summary.score_psi)  # fmt: skip
        return summary


def model_dir_from(models: str) -> Path:
    """Thư mục mô hình: --models, hoặc tải từ MLflow nếu đặt MODEL_URI."""
    uri = os.environ.get("MODEL_URI")
    if not uri:
        return Path(models)
    from riskops.registry import fetch

    tracking = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    return fetch(tracking, uri, Path(tempfile.mkdtemp(prefix="fraud-model-")))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--brokers", default=os.environ.get("BROKERS", "localhost:19092"))
    ap.add_argument("--redis", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--models", default="models")
    ap.add_argument("--topic", default="transactions")
    ap.add_argument("--group", default="fraud-scorer")
    ap.add_argument("--max-events", type=int, default=None, help="dừng sau N sự kiện và ghi báo cáo")
    ap.add_argument("--report", default="reports/stream.md")
    ap.add_argument("--log-features", default=None)
    ap.add_argument("--metrics-port", type=int, default=8001)
    ap.add_argument("--drift-window", type=int, default=int(os.environ.get("DRIFT_WINDOW", "2000")))
    ap.add_argument("--drift-every", type=int, default=int(os.environ.get("DRIFT_EVERY", "500")))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import redis
    from confluent_kafka import Consumer, Producer

    model_dir = model_dir_from(a.models)
    scorer = Scorer(RedisWindows(redis.Redis.from_url(a.redis)), model_dir)
    drift = DriftMonitor(model_dir, a.drift_window, a.drift_every, publish=bool(a.metrics_port))
    alert = from_env()
    metrics = Metrics(a.metrics_port)
    consumer = Consumer({"bootstrap.servers": a.brokers, "group.id": a.group, "auto.offset.reset": "earliest",
                         "enable.auto.commit": False})  # fmt: skip
    alerts_out = Producer({"bootstrap.servers": a.brokers})
    consumer.subscribe([a.topic])
    flog = open(a.log_features, "w", encoding="utf-8") if a.log_features else None
    lat, n, n_alert, tp, n_fraud, t_first = [], 0, 0, 0, 0, None
    log.info("consumer sẵn sàng, ngưỡng %.3f", scorer.threshold)
    try:
        while a.max_events is None or n < a.max_events:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                log.error("lỗi Kafka: %s", msg.error())
                continue
            tx = json.loads(msg.value())
            res = scorer.score(tx)
            if res["alert"]:
                alert(tx, res["score"])
                alerts_out.produce("alerts", key=str(tx["tx_id"]), value=json.dumps({"tx_id": tx["tx_id"],
                                                                                    "score": res["score"]}))  # fmt: skip
                alerts_out.poll(0)
            consumer.commit(msg, asynchronous=True)
            sent_ms = msg.timestamp()[1]
            latency = time.time() - sent_ms / 1000
            t_first = t_first or time.time()
            n += 1
            lat.append(latency)
            n_alert += res["alert"]
            n_fraud += int(tx.get("is_fraud", 0))
            tp += int(res["alert"] and tx.get("is_fraud", 0))
            metrics.latency.observe(latency)
            metrics.score_time.observe(res["score_ms"] / 1000)
            metrics.events.inc()
            metrics.alerts.inc(res["alert"])
            metrics.labelled.inc(int(tx.get("is_fraud", 0)))
            drift.update(res["features"], res["score"])
            if flog:
                flog.write(json.dumps({"tx_id": tx["tx_id"], **res["features"]}) + "\n")
    finally:
        consumer.close()
        alerts_out.flush(5)
        if flog:
            flog.close()
    if n:
        elapsed = time.time() - t_first
        md = (f"# Chấm điểm trên luồng ({time.strftime('%Y-%m-%d %H:%M')})\n\n"
              f"- {n} sự kiện trong {elapsed:.1f} giây ({n / max(elapsed, 1e-9):.0f} sự kiện/giây đã xử lý)\n"
              f"- Độ trễ mỗi sự kiện (gửi -> chấm xong): p50 {np.percentile(lat, 50) * 1000:.1f} ms, "
              f"p95 {np.percentile(lat, 95) * 1000:.1f} ms, p99 {np.percentile(lat, 99) * 1000:.1f} ms\n"
              f"- Cảnh báo: {n_alert}; trong đó đúng gian lận: {tp}; số gian lận trong luồng: {n_fraud}\n")  # fmt: skip
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(md, encoding="utf-8")
        print(md)


if __name__ == "__main__":
    main()
