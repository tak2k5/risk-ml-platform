"""Fixture dùng chung: một mô hình nhỏ huấn luyện trên dữ liệu giả lập.

Huấn luyện một lần cho cả phiên test (scope="session") để test chạy nhanh.
"""

from __future__ import annotations

import os

os.environ.setdefault("NO_PREFECT", "1")  # test luồng #25 như hàm Python thường, không bật server Prefect

import pytest

from credit_risk.config import Settings
from credit_risk.data import make_synthetic
from credit_risk.train import train


@pytest.fixture(scope="session")
def synthetic_df():
    return make_synthetic(4000, seed=7)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, synthetic_df):
    """Huấn luyện vào thư mục tạm, trả về (settings, metadata)."""
    root = tmp_path_factory.mktemp("run")
    s = Settings(
        model_dir=root / "models",
        reports_dir=root / "reports",
        mlflow_uri=f"sqlite:///{root / 'mlflow.db'}",
        mlflow_artifacts=root / "mlartifacts",
    )
    meta = train(synthetic_df, s, n_trials=0, data_source="synthetic-test", use_mlflow=False)
    return s, meta
