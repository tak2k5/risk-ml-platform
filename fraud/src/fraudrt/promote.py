"""Đưa mô hình gian lận mới (challenger) lên champion trong MLflow dùng chung, nếu nó tốt hơn rõ ràng.

    MLFLOW_TRACKING_URI=http://localhost:5000 python -m fraudrt.promote --holdout models/test_features.parquet

So sánh cả hai trên CÙNG một tập: phần test (15% thời gian cuối) của lần train mới nhất. Champion được train trên dữ
liệu cũ hơn nên cũng chưa thấy đoạn này. Chỉ thay khi PR-AUC tăng ít nhất `--min-gain` (mặc định 0.005), cùng quy
tắc với mô hình tín dụng (credit/pipelines/promote.py), để không thay qua thay lại vì nhiễu.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score

from .train import REGISTERED_MODEL

log = logging.getLogger("fraudrt.promote")


@dataclass
class Decision:
    promote: bool
    champion_pr_auc: float | None
    challenger_pr_auc: float
    reason: str


def decide(champion: float | None, challenger: float, min_gain: float = 0.005) -> Decision:
    """Hàm thuần, dễ test."""
    if champion is None:
        return Decision(True, None, challenger, "chưa có champion")
    gain = challenger - champion
    if gain >= min_gain:
        return Decision(True, champion, challenger, f"PR-AUC tăng {gain:+.4f} >= {min_gain}")
    return Decision(False, champion, challenger, f"PR-AUC chỉ thay đổi {gain:+.4f} < {min_gain}")


def pr_auc_of(model_dir: Path, holdout: pd.DataFrame) -> float:
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
    return float(average_precision_score(holdout["is_fraud"], booster.predict(holdout[meta["features"]])))


def promote(
    tracking_uri: str, holdout: pd.DataFrame, min_gain: float = 0.005, name: str = REGISTERED_MODEL
) -> Decision:
    from riskops.registry import fetch, set_alias, version_of

    challenger_v = version_of(tracking_uri, name, "challenger")
    if challenger_v is None:
        raise SystemExit(f"{name} chưa có phiên bản challenger. Chạy: python -m fraudrt.train --mlflow")
    champion_v = version_of(tracking_uri, name, "champion")
    tmp = Path(tempfile.mkdtemp(prefix="fraud-promote-"))
    challenger = pr_auc_of(fetch(tracking_uri, f"models:/{name}/{challenger_v}", tmp / "challenger"), holdout)
    champion = None
    if champion_v is not None and champion_v != challenger_v:
        champion = pr_auc_of(fetch(tracking_uri, f"models:/{name}/{champion_v}", tmp / "champion"), holdout)
    d = decide(champion, challenger, min_gain)
    if d.promote:
        set_alias(tracking_uri, name, "champion", challenger_v)
    log.info("champion v%s, challenger v%s: %s -> %s", champion_v, challenger_v, d.reason,
             "LÊN CHAMPION" if d.promote else "giữ nguyên")  # fmt: skip
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="models/test_features.parquet", help="đặc trưng + is_fraud của tập test mới")
    ap.add_argument("--min-gain", type=float, default=0.005)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    d = promote(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"), pd.read_parquet(a.holdout), a.min_gain)
    print(json.dumps(d.__dict__, ensure_ascii=False))


if __name__ == "__main__":
    main()
