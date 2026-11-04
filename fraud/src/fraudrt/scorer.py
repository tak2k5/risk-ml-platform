"""Chấm điểm một giao dịch: đặc trưng (Redis) -> LightGBM -> có cảnh báo không. Không phụ thuộc Kafka để test được."""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .features import featurize


class Scorer:
    def __init__(self, store, model_dir: str | Path = "models", threshold: float | None = None):
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        self.booster = lgb.Booster(model_file=str(model_dir / "model.txt"))
        self.features = meta["features"]
        self.threshold = meta["threshold"] if threshold is None else threshold
        self.store = store

    def score(self, tx: dict) -> dict:
        t0 = time.perf_counter()
        f = featurize(self.store, tx)
        x = np.array([[f[c] for c in self.features]], dtype=np.float64)
        p = float(self.booster.predict(x, num_threads=1)[0])
        return {"tx_id": int(tx["tx_id"]), "score": p, "alert": p >= self.threshold, "features": f,
                "score_ms": (time.perf_counter() - t0) * 1000}  # fmt: skip
