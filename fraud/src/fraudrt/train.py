"""Huấn luyện LightGBM trên đặc trưng tính đúng thời điểm, chia train / valid / test THEO THỜI GIAN (70/15/15).

    python -m fraudrt.train --data data/paysim.csv          # PaySim thật (scripts/download_data.sh)
    python -m fraudrt.train --synthetic                       # giả lập, chạy thử
- PR-AUC (average precision) là chỉ số chính: gian lận rất hiếm (PaySim khoảng 0,13%), ROC-AUC dễ cao giả tạo.
- Ngưỡng cảnh báo chọn trên valid (F1 lớn nhất), rồi báo precision / recall tại ngưỡng đó trên test.
- So thêm mô hình KHÔNG có đặc trưng cửa sổ để thấy cửa sổ đóng góp bao nhiêu.
Kết quả: models/model.txt, models/meta.json, models/reference.parquet (mẫu đặc trưng train để đo drift trên luồng),
reports/train.md. Thêm --mlflow để ghi vào MLflow dùng chung (MLFLOW_TRACKING_URI) và đăng ký `fraud-lgbm` với alias
challenger; `python -m fraudrt.promote` quyết định có lên champion không.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .data import load_paysim, make_synthetic
from .features import FEATURES, ROW_FEATURES, offline_features

REGISTERED_MODEL = os.getenv("REGISTERED_MODEL", "fraud-lgbm")
REFERENCE_ROWS = 5000

PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)  # fmt: skip


def time_split(df: pd.DataFrame, valid: float = 0.15, test: float = 0.15):
    df = df.sort_values(["ts", "tx_id"], kind="stable")
    a, b = df["ts"].quantile(1 - valid - test), df["ts"].quantile(1 - test)
    return df[df.ts <= a], df[(df.ts > a) & (df.ts <= b)], df[df.ts > b]


def best_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def fit_eval(X: pd.DataFrame, y: pd.Series, parts, cols: list[str]) -> dict:
    tr, va, te = parts
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X.loc[tr, cols], y[tr], eval_set=[(X.loc[va, cols], y[va])], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(50, verbose=False)])  # fmt: skip
    p_va = model.predict_proba(X.loc[va, cols])[:, 1]
    p_te = model.predict_proba(X.loc[te, cols])[:, 1]
    thr = best_threshold(y[va], p_va)
    pred = p_te >= thr
    tp = int((pred & (y[te] == 1)).sum())
    return {
        "model": model,
        "threshold": thr,
        "PR-AUC": average_precision_score(y[te], p_te),
        "ROC-AUC": roc_auc_score(y[te], p_te),
        "precision": tp / max(int(pred.sum()), 1),
        "recall": tp / max(int(y[te].sum()), 1),
        "cảnh báo": int(pred.sum()),
    }


def run(df: pd.DataFrame, out_models="models", out_reports="reports", label="PaySim") -> dict:
    t0 = time.perf_counter()
    X = offline_features(df)
    feat_s = time.perf_counter() - t0
    y = df.set_index("tx_id").loc[X.index, "is_fraud"]
    tr, va, te = (p["tx_id"].to_numpy() for p in time_split(df))
    full = fit_eval(X, y, (tr, va, te), FEATURES)
    no_win = fit_eval(X, y, (tr, va, te), ROW_FEATURES)
    base = float(y[te].mean())

    Path(out_models).mkdir(parents=True, exist_ok=True)
    full["model"].booster_.save_model(str(Path(out_models) / "model.txt"))
    meta = {"features": FEATURES, "threshold": full["threshold"], "trained_on": label}
    (Path(out_models) / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    # Mẫu đặc trưng + điểm số trên tập train: mốc để consumer đo drift (PSI) trên cửa sổ trượt của luồng
    ref = X.loc[tr, FEATURES].sample(min(REFERENCE_ROWS, len(tr)), random_state=0)
    ref.assign(score=full["model"].predict_proba(ref)[:, 1]).to_parquet(Path(out_models) / "reference.parquet")

    rows = []
    for name, r in (("LightGBM + đặc trưng cửa sổ", full), ("LightGBM, không cửa sổ", no_win)):
        rows.append({"mô hình": name, **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()
                                        if k != "model"}})  # fmt: skip
    imp = pd.Series(full["model"].booster_.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    md = [
        f"# Huấn luyện ({label}, {time.strftime('%Y-%m-%d %H:%M')})",
        "",
        f"{len(df):,} giao dịch, gian lận {df.is_fraud.mean():.3%}. Chia theo thời gian: train {len(tr):,}, valid "
        f"{len(va):,}, test {len(te):,}. PR-AUC của đoán ngẫu nhiên = tỉ lệ gian lận trong test = {base:.4f}.",
        f"Tính đặc trưng (phát lại toàn bộ lịch sử): {feat_s:.0f} giây.",
        "",
        pd.DataFrame(rows).to_markdown(index=False),
        "",
        "Ngưỡng chọn trên valid (F1 lớn nhất); precision / recall / số cảnh báo đo trên test.",
        "",
        "Đặc trưng quan trọng nhất (gain): " + ", ".join(f"{k} ({v / imp.sum():.0%})" for k, v in imp.head(6).items()),
    ]
    text = "\n".join(md) + "\n"
    Path(out_reports).mkdir(parents=True, exist_ok=True)
    (Path(out_reports) / "train.md").write_text(text, encoding="utf-8")
    X.loc[te].assign(is_fraud=y[te]).to_parquet(Path(out_models) / "test_features.parquet")  # để kiểm tra lệch online
    print(text)
    return {"full": full, "no_window": no_win, "baseline": base}


def log_to_mlflow(res: dict, out_models: str | Path, label: str, tracking_uri: str) -> str:
    """Ghi lần chạy vào MLflow dùng chung (cùng chỗ với mô hình tín dụng), đăng ký phiên bản mới alias challenger."""
    from riskops.registry import log_and_register

    full, no_win = res["full"], res["no_window"]
    metrics = {
        "test_pr_auc": full["PR-AUC"],
        "test_roc_auc": full["ROC-AUC"],
        "test_precision": full["precision"],
        "test_recall": full["recall"],
        "test_pr_auc_no_window": no_win["PR-AUC"],
        "test_pr_auc_random": res["baseline"],
    }
    params = {"trained_on": label, "threshold": round(full["threshold"], 6), **PARAMS}
    files = [Path(out_models) / f for f in ("model.txt", "meta.json", "reference.parquet")]
    return log_and_register(tracking_uri, "fraud-realtime", REGISTERED_MODEL, files, params, metrics,
                            run_name=f"lgbm-{time.strftime('%Y%m%d%H%M%S')}",
                            artifact_root=Path(out_models).parent / "mlartifacts")  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/paysim.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--models", default="models")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--mlflow", action="store_true", help="ghi vào MLflow (MLFLOW_TRACKING_URI) và đăng ký challenger")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df, label = (make_synthetic(), "dữ liệu GIẢ LẬP") if a.synthetic else (load_paysim(a.data), "PaySim")
    res = run(df, a.models, a.reports, label)
    if a.mlflow:
        uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        print(f"MLflow: {REGISTERED_MODEL} phiên bản {log_to_mlflow(res, a.models, label, uri)} (challenger)")


if __name__ == "__main__":
    main()
