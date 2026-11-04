"""Tạo đặc trưng (feature engineering).

Mọi phép biến đổi nằm trong một transformer của scikit-learn, nên:
- lúc train và lúc phục vụ API dùng CHUNG một đoạn code (tránh training-serving skew);
- transformer được lưu cùng mô hình trong một file joblib.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import CATEGORICAL_COLS, ENGINEERED_COLS, NUMERIC_COLS
from .data import DAYS_EMPLOYED_ANOMALY


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Chia an toàn: mẫu số bằng 0 hoặc thiếu thì trả NaN thay vì lỗi/vô cực."""
    b = b.replace(0, np.nan)
    return a / b


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Thêm các đặc trưng tỉ lệ dễ giải thích với nghiệp vụ tín dụng.

    Không "học" gì từ dữ liệu (fit không làm gì), nên không gây rò rỉ dữ liệu.
    """

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - giao diện sklearn
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # API có thể gửi null -> pandas để kiểu object; ép về số để LightGBM nhận được
        for col in NUMERIC_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype(float)

        # Hưu trí/không đi làm được mã hóa 365243 ngày: biến thành cờ riêng + NaN
        anomaly = X["DAYS_EMPLOYED"] == DAYS_EMPLOYED_ANOMALY
        X["EMPLOYED_ANOMALY"] = anomaly.astype(int)
        days_employed = X["DAYS_EMPLOYED"].where(~anomaly, np.nan)

        X["CREDIT_INCOME_RATIO"] = _safe_div(X["AMT_CREDIT"], X["AMT_INCOME_TOTAL"])
        X["ANNUITY_INCOME_RATIO"] = _safe_div(X["AMT_ANNUITY"], X["AMT_INCOME_TOTAL"])
        X["CREDIT_TERM"] = _safe_div(X["AMT_CREDIT"], X["AMT_ANNUITY"])  # số kỳ trả xấp xỉ
        X["AGE_YEARS"] = -X["DAYS_BIRTH"] / 365.25
        X["EMPLOYED_YEARS"] = -days_employed / 365.25
        X["EXT_SOURCE_MEAN"] = X[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)

        # Cột phân loại: NaN -> "Unknown" để OneHotEncoder xử lý thống nhất
        for col in CATEGORICAL_COLS:
            X[col] = X[col].fillna("Unknown").astype(str)

        return X[NUMERIC_COLS + ENGINEERED_COLS + CATEGORICAL_COLS]

    def get_feature_names_out(self, input_features=None):
        return np.array(NUMERIC_COLS + ENGINEERED_COLS + CATEGORICAL_COLS)
