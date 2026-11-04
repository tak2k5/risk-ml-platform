"""Giải thích từng quyết định bằng SHAP.

Ngân hàng cần nói được với khách: "Hồ sơ bị từ chối chủ yếu vì A, B, C".
SHAP chia "điểm rủi ro" của một hồ sơ thành phần đóng góp của từng đặc trưng:
- giá trị dương: đẩy rủi ro LÊN;
- giá trị âm: kéo rủi ro XUỐNG.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Tên dễ đọc cho các đặc trưng quan trọng (phần còn lại giữ tên gốc)
READABLE = {
    "EXT_SOURCE_MEAN": "Điểm tín dụng ngoài (trung bình)",
    "EXT_SOURCE_1": "Điểm tín dụng ngoài 1",
    "EXT_SOURCE_2": "Điểm tín dụng ngoài 2",
    "EXT_SOURCE_3": "Điểm tín dụng ngoài 3",
    "ANNUITY_INCOME_RATIO": "Tỉ lệ tiền trả hằng kỳ / thu nhập",
    "CREDIT_INCOME_RATIO": "Tỉ lệ khoản vay / thu nhập",
    "CREDIT_TERM": "Số kỳ trả nợ",
    "AGE_YEARS": "Tuổi",
    "EMPLOYED_YEARS": "Số năm đi làm",
    "EMPLOYED_ANOMALY": "Không đi làm / hưu trí",
    "AMT_INCOME_TOTAL": "Thu nhập",
    "AMT_CREDIT": "Số tiền vay",
    "AMT_ANNUITY": "Tiền trả hằng kỳ",
    "REGION_RATING_CLIENT": "Xếp hạng vùng",
    "DAYS_BIRTH": "Tuổi (số ngày từ khi sinh)",
    "DAYS_EMPLOYED": "Thời gian đi làm",
    "DAYS_ID_PUBLISH": "Thời gian từ lần cấp giấy tờ",
    "CNT_CHILDREN": "Số con",
}

# Cột phân loại sau one-hot có dạng "OCCUPATION_TYPE_Laborers" -> "Nghề nghiệp = Laborers"
CATEGORY_LABELS = {
    "NAME_CONTRACT_TYPE": "Loại hợp đồng",
    "FLAG_OWN_CAR": "Có ô tô",
    "FLAG_OWN_REALTY": "Có bất động sản",
    "NAME_INCOME_TYPE": "Nguồn thu nhập",
    "NAME_EDUCATION_TYPE": "Học vấn",
    "NAME_FAMILY_STATUS": "Tình trạng hôn nhân",
    "NAME_HOUSING_TYPE": "Nhà ở",
    "OCCUPATION_TYPE": "Nghề nghiệp",
}


def readable(feature: str) -> str:
    """Đổi tên cột kỹ thuật thành nhãn tiếng Việt cho khách hàng/nhân viên tín dụng đọc."""
    if feature in READABLE:
        return READABLE[feature]
    for col, label in CATEGORY_LABELS.items():
        if feature.startswith(col + "_"):
            return f"{label} = {feature[len(col) + 1 :]}"
    return feature


@dataclass
class Reason:
    feature: str
    label: str
    contribution: float  # đơn vị log-odds; dương = tăng rủi ro

    def to_dict(self) -> dict:
        return {"feature": self.feature, "label": self.label, "contribution": round(self.contribution, 4)}


class Explainer:
    """Bọc SHAP để dùng lại (khởi tạo một lần khi API khởi động, vì khá tốn thời gian)."""

    def __init__(self, pipeline: Pipeline):
        import shap  # import muộn: test nào không cần SHAP sẽ chạy nhanh hơn

        self.pipeline = pipeline
        self.model = pipeline.named_steps["model"]
        self._transform = Pipeline(pipeline.steps[:-1])  # mọi bước trừ mô hình
        if hasattr(self.model, "booster_"):
            self.kind = "tree"
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.kind = "linear"
            self.explainer = None

    def contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """Trả về bảng đóng góp SHAP: mỗi dòng một hồ sơ, mỗi cột một đặc trưng."""
        Xt = self._transform.transform(X)
        if self.kind == "tree":
            values = self.explainer.shap_values(Xt)
            if isinstance(values, list):  # một số phiên bản SHAP trả list [lớp 0, lớp 1]
                values = values[1]
        else:
            # Mô hình tuyến tính: đóng góp = hệ số x (giá trị - trung bình); dữ liệu đã chuẩn hóa nên trung bình ~ 0
            values = Xt.to_numpy() * self.model.coef_[0]
        return pd.DataFrame(np.asarray(values), columns=Xt.columns, index=X.index)

    def top_reasons(self, X: pd.DataFrame, k: int = 3) -> list[list[Reason]]:
        """k lý do làm tăng rủi ro nhiều nhất cho từng hồ sơ."""
        contrib = self.contributions(X)
        result: list[list[Reason]] = []
        for _, row in contrib.iterrows():
            top = row.sort_values(ascending=False).head(k)
            result.append(
                [Reason(feature=f, label=readable(f), contribution=float(v)) for f, v in top.items() if v > 0]
            )
        return result


def global_importance(explainer: Explainer, X_sample: pd.DataFrame) -> pd.Series:
    """Độ quan trọng tổng thể = trung bình |SHAP| trên một mẫu dữ liệu."""
    contrib = explainer.contributions(X_sample)
    return contrib.abs().mean().sort_values(ascending=False)
