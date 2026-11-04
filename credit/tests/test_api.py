import pytest
from fastapi.testclient import TestClient

from credit_risk.schemas import Application

EXAMPLE = Application.model_config["json_schema_extra"]["example"]


@pytest.fixture()
def client(trained, monkeypatch):
    s, _ = trained
    monkeypatch.setenv("MODEL_DIR", str(s.model_dir))
    from credit_risk.api import app

    with TestClient(app) as c:  # "with" để chạy lifespan (nạp mô hình)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_decision_and_reasons(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["decision"] in {"approve", "reject"}
    assert len(body["reasons"]) <= 3
    assert "X-Process-Time-Ms" in r.headers


def test_predict_rejects_invalid_input(client):
    bad = {**EXAMPLE, "AMT_INCOME_TOTAL": -5}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422  # Pydantic chặn trước khi tới mô hình


def test_batch(client):
    r = client.post("/predict/batch", json={"applications": [EXAMPLE, EXAMPLE]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_high_risk_profile_scores_higher(client):
    risky = {**EXAMPLE, "EXT_SOURCE_2": 0.05, "EXT_SOURCE_3": 0.05, "AMT_ANNUITY": 90000}
    safe = {**EXAMPLE, "EXT_SOURCE_2": 0.9, "EXT_SOURCE_3": 0.9, "AMT_ANNUITY": 10000}
    p_risky = client.post("/predict", json=risky).json()["probability_default"]
    p_safe = client.post("/predict", json=safe).json()["probability_default"]
    assert p_risky > p_safe


def test_metrics_cho_prometheus_gom_drift_tren_cua_so(trained, synthetic_df, monkeypatch):
    """Sau đủ số hồ sơ, /metrics có PSI từng đặc trưng (riskops.StreamDrift) với nhãn model="credit"."""
    s, _ = trained
    monkeypatch.setenv("MODEL_DIR", str(s.model_dir))
    monkeypatch.setenv("DRIFT_WINDOW", "200")
    monkeypatch.setenv("DRIFT_EVERY", "100")
    monkeypatch.setenv("DRIFT_MIN_EVENTS", "100")
    from credit_risk.api import app
    from credit_risk.config import RAW_COLS

    apps = synthetic_df[RAW_COLS].head(200).astype(object).where(synthetic_df[RAW_COLS].head(200).notna(), None)
    with TestClient(app) as c:
        r = c.post("/predict/batch", json={"applications": apps.to_dict("records")})
        assert r.status_code == 200, r.text
        text = c.get("/metrics/").text
    assert 'credit_decisions_total{decision="approve"}' in text
    assert "credit_predict_seconds_bucket" in text
    assert 'risk_feature_psi{feature="AMT_CREDIT",model="credit"}' in text
    assert 'risk_drift_alert{model="credit"}' in text
