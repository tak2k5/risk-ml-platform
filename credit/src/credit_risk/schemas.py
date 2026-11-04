"""Schema dữ liệu vào/ra của API (Pydantic tự kiểm tra kiểu và báo lỗi 422 khi sai)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Application(BaseModel):
    """Một hồ sơ vay. Tên trường giữ nguyên như bộ Home Credit để khớp với pipeline."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "CNT_CHILDREN": 1,
                "AMT_INCOME_TOTAL": 180000,
                "AMT_CREDIT": 600000,
                "AMT_ANNUITY": 28000,
                "AMT_GOODS_PRICE": 540000,
                "DAYS_BIRTH": -12000,
                "DAYS_EMPLOYED": -2500,
                "DAYS_ID_PUBLISH": -3000,
                "EXT_SOURCE_1": None,
                "EXT_SOURCE_2": 0.55,
                "EXT_SOURCE_3": 0.48,
                "REGION_RATING_CLIENT": 2,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "FLAG_OWN_CAR": "N",
                "FLAG_OWN_REALTY": "Y",
                "NAME_INCOME_TYPE": "Working",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                "NAME_FAMILY_STATUS": "Married",
                "NAME_HOUSING_TYPE": "House / apartment",
                "OCCUPATION_TYPE": "Laborers",
            }
        }
    )

    CNT_CHILDREN: int = Field(ge=0, le=20)
    AMT_INCOME_TOTAL: float = Field(gt=0, description="Thu nhập năm")
    AMT_CREDIT: float = Field(gt=0, description="Số tiền vay")
    AMT_ANNUITY: float | None = Field(default=None, gt=0, description="Tiền trả mỗi kỳ")
    AMT_GOODS_PRICE: float | None = Field(default=None, gt=0)
    DAYS_BIRTH: int = Field(lt=0, description="Số ngày từ ngày sinh đến ngày nộp hồ sơ (số âm)")
    DAYS_EMPLOYED: int = Field(description="Số ngày đi làm (số âm); 365243 = không đi làm")
    DAYS_ID_PUBLISH: int = Field(le=0)
    EXT_SOURCE_1: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(default=None, ge=0, le=1)
    REGION_RATING_CLIENT: int = Field(ge=1, le=3)
    NAME_CONTRACT_TYPE: str
    FLAG_OWN_CAR: Literal["Y", "N"]
    FLAG_OWN_REALTY: Literal["Y", "N"]
    NAME_INCOME_TYPE: str
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    OCCUPATION_TYPE: str | None = None


class ReasonOut(BaseModel):
    feature: str
    label: str
    contribution: float


class Prediction(BaseModel):
    probability_default: float = Field(description="Xác suất vỡ nợ ước tính")
    decision: Literal["approve", "reject"]
    threshold: float
    reasons: list[ReasonOut] = Field(description="Các yếu tố làm tăng rủi ro nhiều nhất")
    model_version: str


class BatchRequest(BaseModel):
    applications: list[Application] = Field(min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
