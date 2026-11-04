"""Huấn luyện, đánh giá và lưu mô hình.

Chạy:
    python -m credit_risk.train                       # dữ liệu thật trong data/
    python -m credit_risk.train --synthetic 20000     # dữ liệu giả lập, không cần Kaggle
    python -m credit_risk.train --tune 30             # tinh chỉnh LightGBM bằng Optuna 30 lần thử

Quy trình (giống Phần 3.2 của giáo trình):
    1. Chia train / val / test (test để riêng, chỉ dùng MỘT lần ở cuối).
    2. Baseline Logistic Regression.
    3. LightGBM (có thể tinh chỉnh bằng Optuna trên val).
    4. Chọn mô hình có AUC val cao hơn, chọn ngưỡng (vẫn trên val).
    5. Đánh giá cả hai mô hình trên test, lưu mô hình được chọn + metadata + báo cáo.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import CATEGORICAL_COLS, NUMERIC_COLS, Settings, get_settings
from .data import load_raw, make_synthetic, split_xy
from .evaluate import choose_threshold, compute_metrics
from .pipeline import build_pipeline

log = logging.getLogger("credit_risk.train")


def split_data(df: pd.DataFrame, seed: int) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Chia 70% train, 15% val, 15% test, giữ nguyên tỉ lệ vỡ nợ ở mỗi phần (stratify)."""
    X, y = split_xy(df)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    return {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}


def tune_lgbm(splits, n_trials: int, seed: int) -> dict[str, Any]:
    """Tìm tham số LightGBM tốt nhất theo ROC-AUC trên val."""
    import optuna
    from sklearn.metrics import roc_auc_score

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 127, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        pipe = build_pipeline("lgbm", params, seed=seed).fit(X_train, y_train)
        return roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Optuna: AUC val tốt nhất %.4f với %s", study.best_value, study.best_params)
    return study.best_params


def train(
    df: pd.DataFrame,
    settings: Settings | None = None,
    n_trials: int = 0,
    data_source: str = "unknown",
    use_mlflow: bool = True,
) -> dict[str, Any]:
    """Toàn bộ quy trình huấn luyện. Trả về metadata (cũng được ghi ra file)."""
    s = settings or get_settings()
    s.model_dir.mkdir(parents=True, exist_ok=True)
    s.reports_dir.mkdir(parents=True, exist_ok=True)

    splits = split_data(df, s.seed)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    log.info(
        "Train %d | Val %d | Test %d | tỉ lệ vỡ nợ %.2f%%", len(X_train), len(X_val), len(X_test), 100 * y_train.mean()
    )

    # 1) Baseline
    baseline = build_pipeline("logreg", seed=s.seed).fit(X_train, y_train)

    # 2) LightGBM (tinh chỉnh nếu được yêu cầu)
    lgbm_params = tune_lgbm(splits, n_trials, s.seed) if n_trials > 0 else {}
    lgbm = build_pipeline("lgbm", lgbm_params, seed=s.seed).fit(X_train, y_train)
    candidates = {"logreg": baseline, "lgbm": lgbm}

    # 3) Chọn mô hình theo ROC-AUC trên VAL (không nhìn test), rồi chọn ngưỡng theo chi phí cũng trên VAL
    val_auc = {name: roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]) for name, m in candidates.items()}
    selected = max(val_auc, key=val_auc.get)
    model = candidates[selected]
    log.info("AUC val: %s -> chọn %s", {k: round(v, 4) for k, v in val_auc.items()}, selected)
    thr = choose_threshold(y_val, model.predict_proba(X_val)[:, 1], cost_fn=s.cost_fn, cost_fp=s.cost_fp)

    # 4) Đánh giá trên TEST đúng một lần
    results = {name: compute_metrics(y_test, m.predict_proba(X_test)[:, 1]).to_dict() for name, m in candidates.items()}
    log.info("Test: baseline AUC %.4f | LightGBM AUC %.4f", results["logreg"]["roc_auc"], results["lgbm"]["roc_auc"])

    # 5) Lưu mô hình + metadata + mẫu dữ liệu tham chiếu (để #25 so sánh drift về sau)
    joblib.dump(model, s.model_path)
    X_train.sample(min(5000, len(X_train)), random_state=s.seed).to_csv(s.reference_path, index=False)
    metadata = {
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "n_rows": int(len(df)),
        "threshold": thr.threshold,
        "threshold_detail": thr.__dict__,
        "cost_fn": s.cost_fn,
        "cost_fp": s.cost_fp,
        "selected_model": selected,
        "val_roc_auc": {k: round(v, 4) for k, v in val_auc.items()},
        "lgbm_params": lgbm_params,
        "test_metrics": results,
        "features": {"numeric": NUMERIC_COLS, "categorical": CATEGORICAL_COLS},
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }
    s.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    write_report(metadata, s.reports_dir / "metrics.md")

    if use_mlflow:
        try:
            metadata["registry_version"] = log_to_mlflow(s, metadata, s.model_path)
        except Exception as exc:  # MLflow lỗi không được làm hỏng lần train
            log.warning("Ghi MLflow thất bại (%s). Mô hình vẫn được lưu ở %s", exc, s.model_path)
    return metadata


def write_report(metadata: dict[str, Any], path: Path) -> None:
    """Bảng kết quả dạng Markdown, dán thẳng vào README."""
    r = metadata["test_metrics"]
    t = metadata["threshold_detail"]
    lines = [
        f"# Kết quả trên tập test ({metadata['data_source']}, {metadata['n_rows']:,} hồ sơ)",
        "",
        "| Mô hình | ROC-AUC | PR-AUC | KS | Brier |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, label in [("logreg", "Logistic Regression (baseline)"), ("lgbm", "LightGBM")]:
        m = r[name]
        lines.append(f"| {label} | {m['roc_auc']:.4f} | {m['pr_auc']:.4f} | {m['ks']:.4f} | {m['brier']:.4f} |")
    lines += [
        "",
        f"Ngưỡng chọn trên val theo chi phí (FN = {metadata['cost_fn']}, FP = {metadata['cost_fp']}): "
        f"**{t['threshold']}**, duyệt {100 * t['approve_rate']:.1f}% hồ sơ, "
        f"chặn được {100 * t['recall_default']:.1f}% người vỡ nợ.",
        "",
        f"Mô hình được chọn (theo AUC trên val): **{metadata['selected_model']}**.",
    ]
    path.write_text("\n".join(lines) + "\n")


def log_to_mlflow(s: Settings, metadata: dict[str, Any], model_path: Path) -> str | None:
    """Ghi lần chạy vào MLflow dùng chung và đăng ký mô hình với alias 'challenger'.

    Code ghi/đăng ký nằm ở `riskops.registry` (mô hình gian lận dùng cùng hàm). Ngoài model.joblib và metadata.json
    còn ghi reference.csv.gz, để API nạp từ registry (`MODEL_URI`) vẫn đo được drift.
    Không có MLflow thì bỏ qua (train vẫn thành công) để notebook đơn giản vẫn chạy được.
    Việc đưa challenger lên champion do pipelines/promote.py quyết định (dự án #25).
    """
    try:
        from riskops.registry import log_and_register
    except ImportError:
        log.warning("Không tìm thấy riskops (thêm ../riskops/src vào PYTHONPATH), bỏ qua bước ghi MLflow.")
        return None

    metrics = {f"test_{name}_{k}": v for name, ms in metadata["test_metrics"].items() for k, v in ms.items()}
    params = {
        "data_source": metadata["data_source"],
        "selected_model": metadata["selected_model"],
        "threshold": metadata["threshold"],
        **metadata["lgbm_params"],
    }
    files = [p for p in (model_path, s.metadata_path, s.reference_path) if p.exists()]
    try:
        return log_and_register(
            s.mlflow_uri,
            s.experiment,
            s.registered_model,
            files,
            params,
            metrics,
            run_name=f"{metadata['selected_model']}-{metadata['model_version']}",
            artifact_root=s.mlflow_artifacts,
        )
    except ImportError:
        log.warning("Chưa cài mlflow, bỏ qua bước ghi MLflow.")
        return None
    except Exception as exc:  # backend không hỗ trợ registry, server tắt...
        log.warning("Không ghi được MLflow: %s", exc)
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình rủi ro tín dụng")
    parser.add_argument("--synthetic", type=int, default=0, help="Dùng N hồ sơ giả lập thay vì dữ liệu thật")
    parser.add_argument("--nrows", type=int, default=None, help="Chỉ đọc N dòng đầu (chạy thử nhanh)")
    parser.add_argument("--tune", type=int, default=0, help="Số lần thử Optuna (0 = không tinh chỉnh)")
    parser.add_argument("--no-mlflow", action="store_true", help="Không ghi MLflow")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = get_settings()
    if args.synthetic:
        df, source = make_synthetic(args.synthetic, seed=s.seed), f"synthetic-{args.synthetic}"
    else:
        df, source = load_raw(s.data_path, nrows=args.nrows), "home-credit"

    meta = train(df, s, n_trials=args.tune, data_source=source, use_mlflow=not args.no_mlflow)
    print(json.dumps(meta["test_metrics"], indent=2))
    print(f"Ngưỡng: {meta['threshold']} | Mô hình lưu tại {s.model_path}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
