"""Luồng tự động: kiểm tra drift -> huấn luyện lại -> so sánh -> thay mô hình nếu tốt hơn.

Chạy bằng Prefect (có giao diện theo dõi, retry, lịch chạy):
    pip install -r requirements-pipelines.txt
    python -m pipelines.flow --synthetic-shift 0.8 --concept-shift 0.8

Không cài Prefect thì luồng vẫn chạy như hàm Python bình thường (decorator được thay bằng hàm rỗng).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from credit_risk.config import RAW_COLS, TARGET, Settings, get_settings
from credit_risk.data import make_synthetic
from credit_risk.train import train

from .drift import DriftSummary
from .monitor import run_monitor
from .promote import Decision, auc_on, compare, publish, set_mlflow_champion

try:  # Prefect là tùy chọn; đặt NO_PREFECT=1 để chạy như hàm thường (dùng trong test)
    if os.getenv("NO_PREFECT") == "1":
        raise ImportError
    from prefect import flow, task
except ImportError:

    def task(fn=None, **_kwargs):
        return fn if fn is not None else (lambda f: f)

    flow = task

log = logging.getLogger("pipelines.flow")

# Hiệu năng giảm quá mức này (so với AUC lúc train) cũng là lý do để retrain, kể cả khi PSI chưa báo động
MAX_AUC_DROP = 0.03


@dataclass
class FlowResult:
    drifted: bool
    champion_auc_now: float | None
    retrained: bool
    decision: dict | None
    candidate_dir: str | None


@task(retries=1)
def check_drift(current: pd.DataFrame, serving_dir: Path) -> DriftSummary:
    return run_monitor(current, serving_dir)


@task
def champion_performance(labeled: pd.DataFrame, serving_dir: Path) -> tuple[float | None, float | None]:
    """AUC của champion trên dữ liệu mới có nhãn, và AUC lúc nó được train (để so độ sụt)."""
    if not (serving_dir / "model.joblib").exists():
        return None, None
    model = joblib.load(serving_dir / "model.joblib")
    meta = json.loads((serving_dir / "metadata.json").read_text())
    trained_auc = meta["test_metrics"][meta.get("selected_model", "lgbm")]["roc_auc"]
    return auc_on(model, labeled[RAW_COLS], labeled[TARGET]), trained_auc


@task
def retrain(train_df: pd.DataFrame, base: Settings, candidate_dir: Path, use_mlflow: bool) -> dict:
    s = Settings(
        model_dir=candidate_dir,
        reports_dir=candidate_dir / "reports",
        mlflow_uri=base.mlflow_uri,
        mlflow_artifacts=base.mlflow_artifacts,
        cost_fn=base.cost_fn,
        cost_fp=base.cost_fp,
    )
    return train(train_df, s, data_source="retrain", use_mlflow=use_mlflow)


@task
def evaluate_and_promote(
    holdout: pd.DataFrame, serving_dir: Path, candidate_dir: Path, base: Settings, registry_version: str | None
) -> Decision:
    champion = joblib.load(serving_dir / "model.joblib") if (serving_dir / "model.joblib").exists() else None
    challenger = joblib.load(candidate_dir / "model.joblib")
    decision = compare(champion, challenger, holdout[RAW_COLS], holdout[TARGET])
    log.info("Quyết định: %s (%s)", "PROMOTE" if decision.promote else "giữ champion", decision.reason)
    if decision.promote:
        publish(candidate_dir, serving_dir)
        set_mlflow_champion(base.registered_model, registry_version, base.mlflow_uri)
    return decision


@flow(name="credit-risk-retrain")
def retrain_flow(
    labeled_recent: pd.DataFrame,
    serving_dir: Path | None = None,
    work_dir: Path | None = None,
    force: bool = False,
    use_mlflow: bool = True,
    seed: int = 42,
) -> FlowResult:
    """`labeled_recent`: dữ liệu gần đây đã có nhãn (ví dụ hồ sơ 1-2 tháng trước, đã biết trả nợ hay không)."""
    base = get_settings()
    serving_dir = serving_dir or base.model_dir
    work_dir = work_dir or (base.model_dir / "candidates")

    drift = check_drift(labeled_recent, serving_dir)
    auc_now, auc_trained = champion_performance(labeled_recent, serving_dir)
    perf_drop = auc_now is not None and auc_trained is not None and (auc_trained - auc_now) > MAX_AUC_DROP
    log.info("Drift=%s | AUC champion hiện tại=%s (lúc train %s)", drift.drifted, auc_now, auc_trained)

    if not (force or drift.drifted or perf_drop):
        return FlowResult(drift.drifted, auc_now, False, None, None)

    # Giữ 20% làm holdout để so champion và challenger một cách công bằng
    train_part, holdout = train_test_split(
        labeled_recent, test_size=0.2, stratify=labeled_recent[TARGET], random_state=seed
    )
    candidate_dir = work_dir / datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    meta = retrain(train_part, base, candidate_dir, use_mlflow)
    decision = evaluate_and_promote(holdout, serving_dir, candidate_dir, base, meta.get("registry_version"))
    return FlowResult(drift.drifted, auc_now, True, asdict(decision), str(candidate_dir))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chạy luồng retrain một lần")
    parser.add_argument("--labeled", type=Path, help="CSV dữ liệu gần đây đã có nhãn TARGET")
    parser.add_argument("--synthetic-shift", type=float, default=0.0)
    parser.add_argument("--concept-shift", type=float, default=0.0)
    parser.add_argument("--force", action="store_true", help="Retrain kể cả khi không có drift")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.labeled:
        df = pd.read_csv(args.labeled)
    else:
        df = make_synthetic(12000, seed=777, covariate_shift=args.synthetic_shift, concept_shift=args.concept_shift)
    result = retrain_flow(df, force=args.force, use_mlflow=not args.no_mlflow)
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
