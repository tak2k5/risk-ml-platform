"""Đo độ lệch phân phối (drift) giữa dữ liệu tham chiếu (lúc train) và dữ liệu hiện tại.

Dùng chung cho hai mô hình của repo:
- tín dụng (credit/pipelines/monitor.py): so cả một lô hồ sơ tháng mới với mẫu train, chạy theo lịch;
- gian lận (fraud/src/fraudrt/consumer.py) và API tín dụng: `StreamDrift` tính PSI trên cửa sổ trượt các sự kiện
  gần nhất, đẩy lên Prometheus để Grafana vẽ và cảnh báo ngay khi luồng thay đổi.

PSI (Population Stability Index), chỉ số quen thuộc trong ngân hàng:
    PSI = sum( (cur% - ref%) * ln(cur% / ref%) )  trên các "ngăn" (bin)
    < 0.10  : ổn định
    0.10-0.25: có thay đổi, cần theo dõi
    > 0.25  : thay đổi lớn, nên xem xét huấn luyện lại

KS test (Kolmogorov-Smirnov) cho cột số: p-value nhỏ nghĩa là hai phân phối khác nhau.
Lưu ý: với dữ liệu rất lớn, KS gần như luôn "có ý nghĩa", nên ta dựa vào PSI để ra quyết định.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PSI_WARN = 0.10
PSI_ALERT = 0.25
_EPS = 1e-4  # tránh chia cho 0 / log(0) khi một ngăn trống


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """PSI cho cột số. Ngăn được cắt theo phân vị của dữ liệu tham chiếu."""
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    if ref.empty or cur.empty:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # cột gần như hằng số
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    return _psi(ref_pct, cur_pct)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    """PSI cho cột phân loại: mỗi giá trị là một ngăn."""
    ref = ref.fillna("Unknown").astype(str)
    cur = cur.fillna("Unknown").astype(str)
    cats = sorted(set(ref) | set(cur))
    ref_pct = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_pct, cur_pct)


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct = np.clip(ref_pct, _EPS, None)
    cur_pct = np.clip(cur_pct, _EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def status(psi: float) -> str:
    if psi >= PSI_ALERT:
        return "alert"
    if psi >= PSI_WARN:
        return "warn"
    return "ok"


@dataclass
class DriftSummary:
    table: pd.DataFrame  # một dòng cho mỗi đặc trưng
    n_alert: int
    n_warn: int
    score_psi: float | None  # PSI của điểm số mô hình (nếu có)

    @property
    def drifted(self) -> bool:
        """Quy tắc đơn giản: có ít nhất một đặc trưng 'alert' hoặc điểm số mô hình lệch mạnh."""
        return self.n_alert > 0 or (self.score_psi is not None and self.score_psi >= PSI_ALERT)


def drift_table(ref: pd.DataFrame, cur: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric:
        r, c = pd.to_numeric(ref[col], errors="coerce"), pd.to_numeric(cur[col], errors="coerce")
        p = psi_numeric(r, c)
        ks_p = float(ks_2samp(r.dropna(), c.dropna()).pvalue) if r.notna().any() and c.notna().any() else 1.0
        rows.append({"feature": col, "type": "numeric", "psi": round(p, 4), "ks_pvalue": round(ks_p, 4)})
    for col in categorical:
        p = psi_categorical(ref[col], cur[col])
        rows.append({"feature": col, "type": "categorical", "psi": round(p, 4), "ks_pvalue": None})
    table = pd.DataFrame(rows)
    table["status"] = table["psi"].map(status)
    return table.sort_values("psi", ascending=False).reset_index(drop=True)


def summarize(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    ref_scores: np.ndarray | None = None,
    cur_scores: np.ndarray | None = None,
) -> DriftSummary:
    table = drift_table(ref, cur, numeric, categorical)
    score_psi = None
    if ref_scores is not None and cur_scores is not None:
        score_psi = round(psi_numeric(pd.Series(ref_scores), pd.Series(cur_scores)), 4)
    return DriftSummary(
        table=table,
        n_alert=int((table["status"] == "alert").sum()),
        n_warn=int((table["status"] == "warn").sum()),
        score_psi=score_psi,
    )


def split_columns(ref: pd.DataFrame, max_levels: int = 10) -> tuple[list[str], list[str]]:
    """Chia cột thành (số, phân loại). Cột số có ít giá trị khác nhau (cờ 0/1, one-hot) coi là phân loại,
    vì PSI theo phân vị không cắt được ngăn cho cột chỉ có hai giá trị."""
    numeric, categorical = [], []
    for col in ref.columns:
        is_num = pd.api.types.is_numeric_dtype(ref[col])
        (numeric if is_num and ref[col].nunique() > max_levels else categorical).append(col)
    return numeric, categorical


class StreamDrift:
    """PSI trên cửa sổ trượt `window` sự kiện gần nhất, so với mẫu tham chiếu lúc train.

        tracker = StreamDrift(reference_df, window=2000, every=500)
        summary = tracker.update(features_dict, score)   # None cho tới khi đủ dữ liệu / tới lượt tính

    Tính lại mỗi `every` sự kiện (không phải mỗi sự kiện) để không làm chậm việc chấm điểm.
    Chỉ bắt đầu khi cửa sổ có ít nhất `min_events` sự kiện: PSI trên vài chục dòng rất nhiễu.
    """

    def __init__(
        self,
        reference: pd.DataFrame,
        window: int = 2000,
        every: int = 500,
        min_events: int | None = None,
        ref_scores: np.ndarray | None = None,
        numeric: list[str] | None = None,
        categorical: list[str] | None = None,
    ):
        if numeric is None or categorical is None:
            numeric, categorical = split_columns(reference)
        self.numeric, self.categorical = numeric, categorical
        self.reference = reference[numeric + categorical].reset_index(drop=True)
        self.ref_scores = ref_scores
        self.rows: deque[dict] = deque(maxlen=window)
        self.scores: deque[float] = deque(maxlen=window)
        self.every = every
        self.min_events = min_events if min_events is not None else min(window, 500)
        self.seen = 0
        self.last: DriftSummary | None = None

    def update(self, row: dict, score: float | None = None) -> DriftSummary | None:
        self.rows.append({c: row.get(c) for c in self.numeric + self.categorical})
        if score is not None:
            self.scores.append(float(score))
        self.seen += 1
        if len(self.rows) < self.min_events or self.seen % self.every:
            return None
        current = pd.DataFrame(list(self.rows))
        use_scores = self.ref_scores is not None and len(self.scores) == len(self.rows)
        self.last = summarize(
            self.reference,
            current,
            self.numeric,
            self.categorical,
            ref_scores=self.ref_scores if use_scores else None,
            cur_scores=np.array(self.scores) if use_scores else None,
        )
        return self.last
