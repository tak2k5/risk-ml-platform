"""Train cả hai mô hình (dữ liệu giả lập hoặc thật), ghi vào MLflow dùng chung, duyệt lên champion.

    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/register.py --synthetic
    MLFLOW_TRACKING_URI=http://localhost:5000 python scripts/register.py      # credit/data, fraud/data thật

Tín dụng: lần đầu chưa có champion thì challenger lên luôn; các lần sau do luồng retrain của credit/pipelines/flow.py
quyết định (so trên tập holdout mới). Gian lận: fraudrt.promote so champion và challenger trên tập test mới nhất.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import pandas as pd
from credit_risk.config import Settings
from credit_risk.data import load_raw, make_synthetic
from credit_risk.train import train
from fraudrt.data import load_paysim
from fraudrt.data import make_synthetic as fraud_synthetic
from fraudrt.promote import promote
from fraudrt.train import log_to_mlflow, run
from pipelines.promote import set_mlflow_champion
from riskops.registry import version_of

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="dữ liệu giả lập (chạy thử, KHÔNG phải kết quả)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uri = os.environ.setdefault("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # báo cáo của lần chạy giả lập ghi riêng vào reports/synthetic/ (không đưa lên GitHub)
    credit_reports = ROOT / "credit/reports" / ("synthetic" if a.synthetic else "")
    s = Settings(model_dir=ROOT / "credit/models", reports_dir=credit_reports, mlflow_uri=uri)
    credit_df = make_synthetic(20000) if a.synthetic else load_raw(s.data_path)
    meta = train(credit_df, s, n_trials=0 if a.synthetic else 30,
                 data_source="synthetic" if a.synthetic else "home-credit", use_mlflow=True)  # fmt: skip
    if version_of(uri, s.registered_model, "champion") is None:
        set_mlflow_champion(s.registered_model, meta.get("registry_version"), uri)

    fraud_dir = ROOT / "fraud/models"
    if a.synthetic:
        df, label = fraud_synthetic(), "dữ liệu GIẢ LẬP"
    else:
        df, label = load_paysim(str(ROOT / "fraud/data/paysim.csv")), "PaySim"
    res = run(df, fraud_dir, ROOT / "fraud/reports" / ("synthetic" if a.synthetic else ""), label)
    log_to_mlflow(res, fraud_dir, label, uri)
    promote(uri, pd.read_parquet(fraud_dir / "test_features.parquet"))

    for name in (s.registered_model, "fraud-lgbm"):
        print(
            f"{name}: champion v{version_of(uri, name, 'champion')}, challenger v{version_of(uri, name, 'challenger')}"
        )


if __name__ == "__main__":
    main()
