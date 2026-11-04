"""Test cho dự án #25 (giám sát drift + retrain)."""

import numpy as np
import pandas as pd

from credit_risk.config import CATEGORICAL_COLS, NUMERIC_COLS
from credit_risk.data import make_synthetic
from pipelines.drift import psi_categorical, psi_numeric, status, summarize
from pipelines.promote import decide


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    same = pd.Series(rng.normal(0, 1, 5000))
    shifted = pd.Series(rng.normal(1, 1, 5000))
    assert psi_numeric(ref, same) < 0.02
    assert psi_numeric(ref, shifted) > 0.25


def test_psi_categorical_detects_new_mix():
    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    cur = pd.Series(["a"] * 30 + ["b"] * 70)
    assert psi_categorical(ref, ref) == 0
    assert status(psi_categorical(ref, cur)) == "alert"


def test_summary_flags_synthetic_covariate_shift():
    ref = make_synthetic(4000, seed=1)
    stable = summarize(ref, make_synthetic(4000, seed=2), NUMERIC_COLS, CATEGORICAL_COLS)
    drifted = summarize(ref, make_synthetic(4000, seed=2, covariate_shift=2.0), NUMERIC_COLS, CATEGORICAL_COLS)
    assert not stable.drifted
    assert drifted.drifted


def test_promotion_rules():
    assert decide(None, 0.70).promote  # chưa có champion
    assert decide(0.75, 0.76).promote  # tốt hơn rõ ràng
    assert not decide(0.75, 0.752).promote  # hơn chút xíu, có thể chỉ là nhiễu
    assert not decide(0.75, 0.70).promote


def test_retrain_flow_promotes_after_concept_shift(trained, tmp_path):
    """Champion train trên dữ liệu cũ; dữ liệu mới bị lệch mạnh -> flow phải retrain và thay mô hình."""
    import shutil

    from pipelines.flow import retrain_flow

    s, _ = trained
    serving = tmp_path / "serving"
    shutil.copytree(s.model_dir, serving)
    new_data = make_synthetic(6000, seed=99, covariate_shift=2.0, concept_shift=1.0)

    result = retrain_flow(new_data, serving_dir=serving, work_dir=tmp_path / "cand", use_mlflow=False)
    assert result.retrained
    assert result.decision["promote"]
