"""Đo drift (PSI, KS): code nằm ở thư viện dùng chung `riskops/src/riskops/drift.py`.

Giữ file này để các import cũ (`from pipelines.drift import ...`) vẫn chạy. Consumer gian lận và API tín dụng dùng
cùng code đó, thêm `StreamDrift` để đo trên cửa sổ trượt của luồng.
"""

from riskops.drift import (  # noqa: F401
    PSI_ALERT,
    PSI_WARN,
    DriftSummary,
    StreamDrift,
    drift_table,
    psi_categorical,
    psi_numeric,
    status,
    summarize,
)
