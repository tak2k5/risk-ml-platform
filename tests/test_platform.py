"""Test xuyên suốt hai mô hình trên MỘT MLflow và MỘT bộ số đo Prometheus (offline, dữ liệu giả lập).

1. Train tín dụng và gian lận, cả hai đăng ký vào cùng registry; duyệt lên champion.
2. API tín dụng và consumer gian lận nạp mô hình thẳng từ registry (`MODEL_URI=models:/<tên>@champion`).
3. Cho lưu lượng chạy qua cả hai: /metrics có PSI drift của cả hai mô hình, cùng tên số đo, khác nhãn `model`.
"""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, generate_latest

mlflow = pytest.importorskip("mlflow")

from credit_risk.config import RAW_COLS, Settings  # noqa: E402
from credit_risk.data import make_synthetic as credit_data  # noqa: E402
from credit_risk.train import train as train_credit  # noqa: E402
from fraudrt.consumer import DriftMonitor, model_dir_from  # noqa: E402
from fraudrt.data import make_synthetic as fraud_data  # noqa: E402
from fraudrt.features import MemoryWindows  # noqa: E402
from fraudrt.promote import promote as promote_fraud  # noqa: E402
from fraudrt.scorer import Scorer  # noqa: E402
from fraudrt.train import log_to_mlflow as log_fraud  # noqa: E402
from fraudrt.train import run as train_fraud  # noqa: E402
from pipelines.promote import set_mlflow_champion  # noqa: E402
from riskops.registry import version_of  # noqa: E402


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """Train cả hai mô hình vào một MLflow (sqlite), đưa cả hai lên champion."""
    root = tmp_path_factory.mktemp("platform")
    uri = f"sqlite:///{root / 'mlflow.db'}"
    s = Settings(model_dir=root / "credit", reports_dir=root / "credit_reports", mlflow_uri=uri,
                 mlflow_artifacts=root / "mlartifacts")  # fmt: skip
    meta = train_credit(credit_data(3000, seed=5), s, n_trials=0, data_source="synthetic-test", use_mlflow=True)
    set_mlflow_champion(s.registered_model, meta["registry_version"], uri)

    fraud = train_fraud(fraud_data(hours=24 * 5, per_hour=120, seed=3), root / "fraud", root / "fraud_reports", "test")
    log_fraud(fraud, root / "fraud", "test", uri)
    assert promote_fraud(uri, pd.read_parquet(root / "fraud" / "test_features.parquet")).promote
    return uri, meta


def test_mot_registry_cho_ca_hai_mo_hinh(registry):
    uri, _ = registry
    names = {m.name for m in mlflow.MlflowClient(tracking_uri=uri).search_registered_models()}
    assert names == {"credit-risk", "fraud-lgbm"}
    assert version_of(uri, "credit-risk", "champion") == "1"
    assert version_of(uri, "fraud-lgbm", "champion") == "1"


def test_hai_dich_vu_nap_champion_tu_registry_va_bao_drift_chung(registry, monkeypatch):
    uri, meta = registry
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    # API tín dụng: nạp credit-risk@champion, chấm 300 hồ sơ của một "tháng mới" đã lệch phân phối
    monkeypatch.setenv("MODEL_URI", "models:/credit-risk@champion")
    monkeypatch.setenv("DRIFT_WINDOW", "300")
    monkeypatch.setenv("DRIFT_EVERY", "150")
    monkeypatch.setenv("DRIFT_MIN_EVENTS", "150")
    from credit_risk.api import app

    month = credit_data(300, seed=11, covariate_shift=2.0)[RAW_COLS]
    month = month.astype(object).where(month.notna(), None)
    with TestClient(app) as c:
        assert c.get("/health").json()["model_version"] == meta["model_version"]
        r = c.post("/predict/batch", json={"applications": month.to_dict("records")})
        assert r.status_code == 200, r.text
        credit_metrics = c.get("/metrics/").text

    # Consumer gian lận: nạp fraud-lgbm@champion (cả mẫu tham chiếu để đo drift), phát lại 2 ngày giao dịch
    monkeypatch.setenv("MODEL_URI", "models:/fraud-lgbm@champion")
    model_dir = model_dir_from("không-dùng-thư-mục-này")
    scorer, drift = Scorer(MemoryWindows(), model_dir), DriftMonitor(model_dir, window=1000, every=500)
    for tx in fraud_data(hours=48, per_hour=120, seed=9).to_dict("records"):
        res = scorer.score(tx)
        drift.update(res["features"], res["score"])

    text = generate_latest(REGISTRY).decode()
    assert 'risk_drift_alert{model="credit"} 1.0' in credit_metrics  # tháng mới lệch mạnh
    assert 'risk_drift_alert{model="fraud"} 0.0' in text  # luồng cùng phân phối với lúc train
    assert 'risk_feature_psi{feature="amount",model="fraud"}' in text
    assert 'risk_feature_psi{feature="AMT_CREDIT",model="credit"}' in text
