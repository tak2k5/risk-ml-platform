"""Đọc dữ liệu thật (Home Credit) hoặc sinh dữ liệu giả lập cùng định dạng.

Vì sao cần dữ liệu giả lập?
- Dữ liệu cuộc thi Kaggle không được phép đưa lên GitHub, nên CI và test không có dữ liệu thật.
- Dữ liệu giả lập có cùng tên cột, cùng kiểu, nên toàn bộ pipeline (train, API, test) chạy được
  mà không cần tài khoản Kaggle. Kết quả báo cáo trong README luôn lấy từ dữ liệu THẬT.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLS, ID_COL, RAW_COLS, TARGET

# Giá trị đặc biệt của Home Credit: 365243 ngày = "không có việc làm / hưu trí"
DAYS_EMPLOYED_ANOMALY = 365243


def load_raw(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Đọc application_train.csv, chỉ giữ các cột dự án dùng (tiết kiệm RAM trên Colab)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Không thấy {path}. Tải dữ liệu theo hướng dẫn trong data/README.md, "
            "hoặc chạy với --synthetic để dùng dữ liệu giả lập."
        )
    usecols = [ID_COL, TARGET, *RAW_COLS]
    df = pd.read_csv(path, usecols=usecols, nrows=nrows)
    return df


def make_synthetic(
    n: int = 20_000,
    seed: int = 42,
    covariate_shift: float = 0.0,
    concept_shift: float = 0.0,
) -> pd.DataFrame:
    """Sinh dữ liệu giả lập giống Home Credit.

    Args:
        n: số hồ sơ.
        seed: hạt giống ngẫu nhiên, cùng seed thì cùng dữ liệu.
        covariate_shift: 0 = giống dữ liệu gốc; càng lớn thì khách càng trẻ, thu nhập càng thấp
            (mô phỏng data drift, dùng ở dự án #25).
        concept_shift: 0 = quan hệ đặc trưng -> vỡ nợ không đổi; càng lớn thì điểm tín dụng ngoài
            càng kém tin cậy, tỉ lệ trả góp/thu nhập càng quan trọng và rủi ro nền tăng
            (mô phỏng concept drift sau cú sốc kinh tế).
    """
    rng = np.random.default_rng(seed)

    age_years = rng.uniform(21, 69, n) - 8 * covariate_shift * rng.uniform(0, 1, n)
    age_years = np.clip(age_years, 20, 70)

    is_pensioner = rng.uniform(0, 1, n) < 0.18
    employed_years = rng.exponential(6, n)
    days_employed = np.where(is_pensioner, DAYS_EMPLOYED_ANOMALY, -(employed_years * 365).astype(int))

    income = rng.lognormal(mean=np.log(160_000) - 0.3 * covariate_shift, sigma=0.5, size=n)
    credit = income * rng.uniform(1, 8, n)
    term_months = rng.uniform(12, 48, n)
    annuity = credit / term_months * 1.1
    goods = credit * rng.uniform(0.8, 1.0, n)

    ext1 = rng.beta(5, 3, n)
    ext2 = rng.beta(5, 3, n)
    ext3 = rng.beta(5, 3, n)
    # Giống dữ liệu thật: EXT_SOURCE_1 thiếu nhiều, EXT_SOURCE_3 thiếu vừa
    ext1 = np.where(rng.uniform(0, 1, n) < 0.5, np.nan, ext1)
    ext3 = np.where(rng.uniform(0, 1, n) < 0.2, np.nan, ext3)

    def pick(options: list[str], probs: list[float]) -> np.ndarray:
        return rng.choice(options, size=n, p=probs)

    df = pd.DataFrame(
        {
            ID_COL: np.arange(100_000, 100_000 + n),
            "CNT_CHILDREN": rng.poisson(0.4, n),
            "AMT_INCOME_TOTAL": income.round(0),
            "AMT_CREDIT": credit.round(0),
            "AMT_ANNUITY": annuity.round(0),
            "AMT_GOODS_PRICE": goods.round(0),
            "DAYS_BIRTH": -(age_years * 365).astype(int),
            "DAYS_EMPLOYED": days_employed,
            "DAYS_ID_PUBLISH": -rng.integers(0, 7000, n),
            "EXT_SOURCE_1": ext1,
            "EXT_SOURCE_2": ext2,
            "EXT_SOURCE_3": ext3,
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], size=n, p=[0.1, 0.74, 0.16]),
            "NAME_CONTRACT_TYPE": pick(["Cash loans", "Revolving loans"], [0.9, 0.1]),
            "FLAG_OWN_CAR": pick(["N", "Y"], [0.66, 0.34]),
            "FLAG_OWN_REALTY": pick(["Y", "N"], [0.69, 0.31]),
            "NAME_INCOME_TYPE": np.where(
                is_pensioner,
                "Pensioner",
                pick(["Working", "Commercial associate", "State servant"], [0.62, 0.28, 0.10]),
            ),
            "NAME_EDUCATION_TYPE": pick(
                ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary"],
                [0.71, 0.24, 0.03, 0.02],
            ),
            "NAME_FAMILY_STATUS": pick(
                ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
                [0.64, 0.15, 0.10, 0.06, 0.05],
            ),
            "NAME_HOUSING_TYPE": pick(
                ["House / apartment", "With parents", "Municipal apartment", "Rented apartment"],
                [0.89, 0.05, 0.04, 0.02],
            ),
            "OCCUPATION_TYPE": pick(
                ["Laborers", "Sales staff", "Core staff", "Managers", "Drivers", "High skill tech staff", None],
                [0.18, 0.10, 0.09, 0.07, 0.06, 0.04, 0.46],
            ),
        }
    )

    # Xác suất vỡ nợ: điểm tín dụng ngoài (EXT_SOURCE) quan trọng nhất, như dữ liệu thật
    ext_mean = np.nanmean(np.vstack([ext1, ext2, ext3]), axis=0)
    annuity_ratio = annuity / income
    logit = (
        -2.9
        # Concept drift: sau cú sốc, điểm tín dụng ngoài (lấy từ trước) kém tin cậy hơn,
        # còn gánh nặng trả nợ / thu nhập trở nên quan trọng hơn nhiều
        - (9.0 - 5.0 * concept_shift) * (ext_mean - 0.62)
        + (2.0 + 6.0 * concept_shift) * (annuity_ratio - 0.2)
        - 0.03 * (age_years - 43)
        + 0.40 * (df["NAME_EDUCATION_TYPE"] == "Lower secondary").to_numpy()
        + 0.35 * (df["REGION_RATING_CLIENT"] == 3).to_numpy()
        + 0.30 * (df["OCCUPATION_TYPE"] == "Laborers").to_numpy()
        + 1.2 * (ext_mean < 0.35)  # phi tuyến: điểm ngoài rất thấp thì rủi ro tăng vọt
        + 0.6 * ((age_years < 28) & (annuity_ratio > 0.3))  # tương tác: trẻ + gánh nợ nặng
        + 0.5 * concept_shift
    )
    prob = 1 / (1 + np.exp(-logit))
    df[TARGET] = (rng.uniform(0, 1, n) < prob).astype(int)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("object")
    return df


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Tách đặc trưng (X) và nhãn (y)."""
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return X, y
