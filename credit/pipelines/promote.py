"""Quyết định có thay mô hình đang chạy (champion) bằng mô hình mới (challenger) hay không.

Nguyên tắc: so sánh cả hai trên CÙNG một tập dữ liệu mới có nhãn mà cả hai chưa từng thấy.
Chỉ thay khi challenger tốt hơn rõ ràng (hơn ít nhất `min_gain` AUC), tránh thay qua thay lại vì nhiễu.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

log = logging.getLogger("pipelines.promote")


@dataclass
class Decision:
    promote: bool
    champion_auc: float | None
    challenger_auc: float
    reason: str


def decide(champion_auc: float | None, challenger_auc: float, min_gain: float = 0.005) -> Decision:
    """Hàm thuần (không đọc/ghi gì), dễ test."""
    if champion_auc is None:
        return Decision(True, None, challenger_auc, "chưa có champion")
    gain = challenger_auc - champion_auc
    if gain >= min_gain:
        return Decision(True, champion_auc, challenger_auc, f"AUC tăng {gain:+.4f} >= {min_gain}")
    return Decision(False, champion_auc, challenger_auc, f"AUC chỉ thay đổi {gain:+.4f} < {min_gain}")


def auc_on(model, X: pd.DataFrame, y: pd.Series) -> float:
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def compare(champion, challenger, X: pd.DataFrame, y: pd.Series, min_gain: float = 0.005) -> Decision:
    champ_auc = auc_on(champion, X, y) if champion is not None else None
    return decide(champ_auc, auc_on(challenger, X, y), min_gain)


def publish(candidate_dir: Path, serving_dir: Path) -> None:
    """Chép mô hình được chọn sang thư mục API đang phục vụ (models/).

    Trong Kubernetes (#26), bước tương đương là build image mới và rollout canary.
    """
    serving_dir.mkdir(parents=True, exist_ok=True)
    for name in ("model.joblib", "metadata.json", "reference.csv.gz"):
        src = candidate_dir / name
        if src.exists():
            shutil.copy2(src, serving_dir / name)
    log.info("Đã publish %s -> %s", candidate_dir, serving_dir)


def set_mlflow_champion(registered_model: str, version: str | None, tracking_uri: str) -> None:
    """Chuyển alias 'champion' sang phiên bản mới trong MLflow Model Registry (nếu có)."""
    if version is None:
        return
    try:
        import mlflow

        mlflow.MlflowClient(tracking_uri=tracking_uri).set_registered_model_alias(registered_model, "champion", version)
        log.info("MLflow: %s@champion -> phiên bản %s", registered_model, version)
    except Exception as exc:  # không có MLflow vẫn chạy được
        log.warning("Không cập nhật được alias champion: %s", exc)


__all__ = ["Decision", "decide", "compare", "publish", "set_mlflow_champion", "auc_on", "np"]
