"""Các thước đo đánh giá và cách chọn ngưỡng theo chi phí.

Với tín dụng, accuracy gần như vô nghĩa (đoán "không vỡ nợ" cho tất cả đã đạt ~92%).
Ta dùng:
- ROC-AUC: khả năng xếp hạng người rủi ro cao lên trước.
- PR-AUC: nhạy hơn với lớp hiếm (vỡ nợ).
- KS: thước đo quen thuộc trong ngân hàng, khoảng cách lớn nhất giữa hai phân phối điểm.
- Brier: xác suất dự đoán có "đúng nghĩa xác suất" không (càng nhỏ càng tốt).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class Metrics:
    roc_auc: float
    pr_auc: float
    ks: float
    brier: float
    default_rate: float

    def to_dict(self) -> dict[str, float]:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Metrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    ks = ks_2samp(y_prob[y_true == 1], y_prob[y_true == 0]).statistic
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
        ks=float(ks),
        brier=float(brier_score_loss(y_true, y_prob)),
        default_rate=float(y_true.mean()),
    )


@dataclass
class ThresholdResult:
    threshold: float
    total_cost: float
    approve_rate: float  # tỉ lệ hồ sơ được duyệt
    recall_default: float  # tỉ lệ người vỡ nợ bị chặn


def choose_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 10.0,
    cost_fp: float = 1.0,
    grid: np.ndarray | None = None,
) -> ThresholdResult:
    """Quét ngưỡng, chọn ngưỡng có tổng chi phí nhỏ nhất.

    Quy ước: xác suất >= ngưỡng thì TỪ CHỐI (dự đoán vỡ nợ).
    - FN (cho vay người sẽ vỡ nợ) tốn cost_fn.
    - FP (từ chối người tốt) tốn cost_fp.
    Chạy trên tập validation, KHÔNG chạy trên test, để test vẫn "sạch".
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    grid = grid if grid is not None else np.linspace(0.01, 0.99, 99)

    best: ThresholdResult | None = None
    for t in grid:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = fn * cost_fn + fp * cost_fp
        result = ThresholdResult(
            threshold=float(round(t, 4)),
            total_cost=float(cost),
            approve_rate=float((y_pred == 0).mean()),
            recall_default=float(tp / (tp + fn)) if (tp + fn) else 0.0,
        )
        if best is None or result.total_cost < best.total_cost:
            best = result
    assert best is not None
    return best
