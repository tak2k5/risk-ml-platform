import numpy as np
import pandas as pd

from credit_risk.config import ENGINEERED_COLS, RAW_COLS
from credit_risk.data import DAYS_EMPLOYED_ANOMALY, make_synthetic
from credit_risk.features import FeatureBuilder


def test_synthetic_has_expected_columns_and_default_rate():
    df = make_synthetic(5000, seed=1)
    assert set(RAW_COLS).issubset(df.columns)
    # Tỉ lệ vỡ nợ gần giống Home Credit (~8%), cho phép dao động
    assert 0.03 < df["TARGET"].mean() < 0.15


def test_synthetic_is_reproducible():
    a = make_synthetic(100, seed=3)
    b = make_synthetic(100, seed=3)
    pd.testing.assert_frame_equal(a, b)


def test_feature_builder_adds_columns_and_handles_anomaly():
    df = make_synthetic(200, seed=2)
    df.loc[df.index[0], "DAYS_EMPLOYED"] = DAYS_EMPLOYED_ANOMALY
    out = FeatureBuilder().fit_transform(df)

    assert set(ENGINEERED_COLS).issubset(out.columns)
    first = out.iloc[0]
    assert first["EMPLOYED_ANOMALY"] == 1
    assert np.isnan(first["EMPLOYED_YEARS"])  # không được biến 365243 thành ~1000 năm đi làm


def test_feature_builder_safe_division_by_zero():
    df = make_synthetic(10, seed=4)
    df["AMT_INCOME_TOTAL"] = 0
    out = FeatureBuilder().fit_transform(df)
    assert out["CREDIT_INCOME_RATIO"].isna().all()
    assert not np.isinf(out.select_dtypes("number").to_numpy(dtype=float)).any()


def test_missing_categorical_becomes_unknown():
    df = make_synthetic(10, seed=5)
    df["OCCUPATION_TYPE"] = None
    out = FeatureBuilder().fit_transform(df)
    assert (out["OCCUPATION_TYPE"] == "Unknown").all()
