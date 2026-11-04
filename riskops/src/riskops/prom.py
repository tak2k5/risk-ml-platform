"""Đẩy kết quả drift lên Prometheus với cùng tên số đo cho cả hai mô hình.

Một dashboard Grafana (monitoring/grafana/risk.json) vẽ được cả tín dụng lẫn gian lận nhờ nhãn `model`:

    risk_feature_psi{model="fraud", feature="amount"}   PSI của từng đặc trưng trên cửa sổ gần nhất
    risk_score_psi{model="credit"}                      PSI của điểm số mô hình
    risk_drift_alert{model="fraud"}                     1 nếu có đặc trưng lệch mạnh (PSI >= 0.25), 0 nếu không
    risk_drift_window_events{model="fraud"}             số sự kiện trong cửa sổ đã dùng để tính
"""

from __future__ import annotations

from prometheus_client import REGISTRY, CollectorRegistry, Gauge

from .drift import DriftSummary


class DriftGauges:
    """Gói các gauge drift của một mô hình. Tạo một lần khi dịch vụ khởi động."""

    def __init__(self, model: str, registry: CollectorRegistry = REGISTRY):
        self.model = model
        self.feature_psi = _gauge(
            "risk_feature_psi", "PSI từng đặc trưng, cửa sổ gần nhất", ["model", "feature"], registry
        )
        self.score_psi = _gauge("risk_score_psi", "PSI của điểm số mô hình", ["model"], registry)
        self.alert = _gauge("risk_drift_alert", "1 nếu phát hiện drift", ["model"], registry)
        self.window = _gauge("risk_drift_window_events", "Số sự kiện trong cửa sổ", ["model"], registry)

    def publish(self, summary: DriftSummary, window_events: int) -> None:
        for _, row in summary.table.iterrows():
            self.feature_psi.labels(self.model, row["feature"]).set(float(row["psi"]))
        if summary.score_psi is not None:
            self.score_psi.labels(self.model).set(summary.score_psi)
        self.alert.labels(self.model).set(1 if summary.drifted else 0)
        self.window.labels(self.model).set(window_events)


def _gauge(name: str, doc: str, labels: list[str], registry: CollectorRegistry) -> Gauge:
    """Lấy lại gauge đã đăng ký nếu có (test tạo nhiều app trong một tiến trình), không thì tạo mới."""
    existing = getattr(registry, "_names_to_collectors", {}).get(name)
    return existing if existing is not None else Gauge(name, doc, labels, registry=registry)
