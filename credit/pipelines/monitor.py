"""Kiểm tra drift: so sánh dữ liệu mới với mẫu dữ liệu lúc train của mô hình đang phục vụ.

    python -m pipelines.monitor --current data/new_month.csv
    python -m pipelines.monitor --synthetic-shift 0.6        # thử với dữ liệu giả lập bị lệch
    python -m pipelines.monitor --synthetic-shift 0.6 --evidently   # thêm báo cáo HTML của Evidently

Kết quả: reports/drift.md (bảng PSI từng đặc trưng) và mã thoát 1 nếu phát hiện drift,
để GitHub Actions / Prefect có thể dựa vào đó mà kích hoạt retrain.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd

from credit_risk.config import CATEGORICAL_COLS, NUMERIC_COLS, RAW_COLS, Settings, get_settings
from credit_risk.data import make_synthetic

from .drift import PSI_ALERT, PSI_WARN, DriftSummary, summarize

log = logging.getLogger("pipelines.monitor")


def load_reference(model_dir: Path) -> pd.DataFrame:
    path = model_dir / "reference.csv.gz"
    if not path.exists():
        raise FileNotFoundError(f"Không có {path}. Hãy train mô hình trước (train.py tự lưu mẫu tham chiếu).")
    return pd.read_csv(path)


def run_monitor(current: pd.DataFrame, model_dir: Path) -> DriftSummary:
    """So sánh `current` với mẫu tham chiếu của mô hình trong `model_dir` (gồm cả drift của điểm số)."""
    ref = load_reference(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    return summarize(
        ref[RAW_COLS],
        current[RAW_COLS],
        NUMERIC_COLS,
        CATEGORICAL_COLS,
        ref_scores=model.predict_proba(ref[RAW_COLS])[:, 1],
        cur_scores=model.predict_proba(current[RAW_COLS])[:, 1],
    )


def write_markdown(summary: DriftSummary, path: Path, title: str = "Báo cáo drift") -> None:
    t = summary.table
    lines = [
        f"# {title}",
        "",
        f"- Đặc trưng lệch mạnh (PSI >= {PSI_ALERT}): **{summary.n_alert}**",
        f"- Đặc trưng cần theo dõi ({PSI_WARN} <= PSI < {PSI_ALERT}): **{summary.n_warn}**",
        f"- PSI của điểm số mô hình: **{summary.score_psi}**",
        f"- Kết luận: **{'CÓ DRIFT, nên retrain' if summary.drifted else 'ổn định'}**",
        "",
        "| Đặc trưng | Loại | PSI | KS p-value | Trạng thái |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in t.iterrows():
        ks = "" if pd.isna(r["ks_pvalue"]) else f"{r['ks_pvalue']:.4f}"
        lines.append(f"| {r['feature']} | {r['type']} | {r['psi']:.4f} | {ks} | {r['status']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_evidently(ref: pd.DataFrame, cur: pd.DataFrame, path: Path) -> bool:
    """Báo cáo HTML trực quan bằng Evidently (tùy chọn). API Evidently đổi giữa các phiên bản nên thử cả hai."""
    try:
        try:  # Evidently >= 0.7
            from evidently import Report
            from evidently.presets import DataDriftPreset

            snapshot = Report([DataDriftPreset()]).run(reference_data=ref, current_data=cur)
            snapshot.save_html(str(path))
        except ImportError:  # Evidently 0.4.x
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=ref, current_data=cur)
            report.save_html(str(path))
        return True
    except Exception as exc:
        log.warning("Không tạo được báo cáo Evidently (%s). Cài: pip install -r requirements-pipelines.txt", exc)
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kiểm tra drift dữ liệu")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--current", type=Path, help="CSV dữ liệu mới (cùng cột với dữ liệu train)")
    src.add_argument("--synthetic-shift", type=float, help="Sinh dữ liệu giả lập với mức covariate shift này")
    parser.add_argument("--evidently", action="store_true", help="Tạo thêm báo cáo HTML bằng Evidently")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s: Settings = get_settings()
    if args.current:
        current = pd.read_csv(args.current)
    else:
        current = make_synthetic(5000, seed=2024, covariate_shift=args.synthetic_shift)

    summary = run_monitor(current, s.model_dir)
    write_markdown(summary, s.reports_dir / "drift.md")
    if args.evidently:
        ref = load_reference(s.model_dir)[RAW_COLS]
        write_evidently(ref, current[RAW_COLS], s.reports_dir / "drift.html")

    print(summary.table.head(10).to_string(index=False))
    print(json.dumps({"drifted": summary.drifted, "n_alert": summary.n_alert, "score_psi": summary.score_psi}))
    return 1 if summary.drifted else 0


if __name__ == "__main__":
    sys.exit(main())
