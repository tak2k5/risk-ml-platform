"""Dựng pipeline scikit-learn: tạo đặc trưng -> tiền xử lý -> mô hình.

Đóng gói mọi bước vào một Pipeline giúp:
- scaler/encoder chỉ học từ tập train (không rò rỉ sang val/test);
- lưu và nạp lại bằng một dòng joblib;
- API chỉ cần gọi pipeline.predict_proba(df).
"""

from __future__ import annotations

from typing import Any, Literal

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL_COLS, ENGINEERED_COLS, NUMERIC_COLS
from .features import FeatureBuilder

ModelKind = Literal["logreg", "lgbm"]

ALL_NUMERIC = NUMERIC_COLS + ENGINEERED_COLS

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 100,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "class_weight": "balanced",  # dữ liệu mất cân bằng (~8% vỡ nợ)
    "verbose": -1,
}


def build_preprocessor(kind: ModelKind) -> ColumnTransformer:
    """Tiền xử lý khác nhau cho từng loại mô hình.

    - Logistic Regression cần điền giá trị thiếu và chuẩn hóa thang đo.
    - LightGBM tự xử lý NaN và không cần chuẩn hóa, nên chỉ mã hóa cột phân loại.
    """
    onehot = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.01, sparse_output=False)

    if kind == "logreg":
        numeric = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                ("scale", StandardScaler()),
            ]
        )
    else:
        numeric = "passthrough"

    return ColumnTransformer(
        [("num", numeric, ALL_NUMERIC), ("cat", onehot, CATEGORICAL_COLS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(kind: ModelKind = "lgbm", params: dict[str, Any] | None = None, seed: int = 42) -> Pipeline:
    """Trả về Pipeline hoàn chỉnh, chưa huấn luyện."""
    if kind == "logreg":
        model = LogisticRegression(max_iter=2000, class_weight="balanced", C=(params or {}).get("C", 1.0))
    elif kind == "lgbm":
        model = LGBMClassifier(**{**DEFAULT_LGBM_PARAMS, **(params or {})}, random_state=seed)
    else:
        raise ValueError(f"Không hỗ trợ loại mô hình: {kind}")

    pipe = Pipeline(
        [
            ("features", FeatureBuilder()),
            ("preprocess", build_preprocessor(kind)),
            ("model", model),
        ]
    )
    # Giữ tên cột qua các bước (cần cho SHAP và để đọc log dễ hơn)
    pipe.set_output(transform="pandas")
    return pipe
