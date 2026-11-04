"""Cấu hình chung của dự án.

Mọi đường dẫn và tham số "cứng" được gom về một chỗ để:
- notebook, script train và API dùng cùng một giá trị;
- muốn đổi chỉ sửa một file (hoặc đặt biến môi trường).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Thư mục gốc của repo (src/credit_risk/config.py -> lùi 3 cấp)
ROOT = Path(__file__).resolve().parents[2]

# Tên cột nhãn và cột định danh trong bộ Home Credit
TARGET = "TARGET"
ID_COL = "SK_ID_CURR"

# Các cột gốc lấy từ application_train.csv.
# Chỉ chọn ~20 cột dễ hiểu để API có schema gọn. Muốn thêm cột: thêm vào đây.
# Lưu ý: KHÔNG dùng CODE_GENDER làm đặc trưng để tránh phân biệt đối xử theo giới tính.
NUMERIC_COLS: list[str] = [
    "CNT_CHILDREN",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_ID_PUBLISH",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "REGION_RATING_CLIENT",
]

CATEGORICAL_COLS: list[str] = [
    "NAME_CONTRACT_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "OCCUPATION_TYPE",
]

RAW_COLS: list[str] = NUMERIC_COLS + CATEGORICAL_COLS

# Đặc trưng tự tạo thêm ở features.py (đều là số)
ENGINEERED_COLS: list[str] = [
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
    "AGE_YEARS",
    "EMPLOYED_YEARS",
    "EMPLOYED_ANOMALY",
    "EXT_SOURCE_MEAN",
]


@dataclass
class Settings:
    """Tham số chạy, đọc được từ biến môi trường (tiện cho Docker và Colab)."""

    data_path: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_PATH", ROOT / "data" / "application_train.csv"))
    )
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", ROOT / "models")))
    reports_dir: Path = field(default_factory=lambda: Path(os.getenv("REPORTS_DIR", ROOT / "reports")))
    mlflow_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}"))
    # Nơi MLflow lưu file mô hình khi chạy local (không có MLflow server)
    mlflow_artifacts: Path = field(
        default_factory=lambda: Path(os.getenv("MLFLOW_ARTIFACTS_DIR", ROOT / "mlartifacts"))
    )
    experiment: str = os.getenv("MLFLOW_EXPERIMENT", "credit-risk")
    registered_model: str = os.getenv("REGISTERED_MODEL", "credit-risk")
    seed: int = int(os.getenv("SEED", "42"))

    # Chi phí giả định để chọn ngưỡng (ghi rõ trong README):
    # cho vay một người sẽ vỡ nợ (FN) mất 10 đơn vị, từ chối nhầm người tốt (FP) mất 1 đơn vị.
    cost_fn: float = float(os.getenv("COST_FN", "10"))
    cost_fp: float = float(os.getenv("COST_FP", "1"))

    @property
    def model_path(self) -> Path:
        return self.model_dir / "model.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "metadata.json"

    @property
    def reference_path(self) -> Path:
        """Mẫu dữ liệu train, làm mốc so sánh drift (dự án #25)."""
        return self.model_dir / "reference.csv.gz"


def get_settings() -> Settings:
    return Settings()
