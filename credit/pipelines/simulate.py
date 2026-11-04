"""Giả lập 12 tháng vận hành để thấy giá trị của giám sát + retrain tự động.

Kịch bản (dữ liệu giả lập, KHÔNG phải số liệu thật):
- Mỗi tháng có một lô hồ sơ mới; khách dần trẻ hơn, thu nhập thấp hơn (covariate shift tăng dần).
- Từ tháng `shock_month` có "cú sốc kinh tế": cùng hồ sơ nhưng rủi ro cao hơn (concept shift).
- So sánh hai cách vận hành:
    static : train một lần rồi để nguyên
    auto   : mỗi tháng chạy retrain_flow (drift / AUC sụt -> retrain -> chỉ thay nếu tốt hơn)

    python -m pipelines.simulate --months 12

Kết quả: reports/sim/simulation.csv, simulation.md, simulation.png
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import joblib
import pandas as pd

from credit_risk.config import RAW_COLS, ROOT, TARGET, Settings
from credit_risk.data import make_synthetic
from credit_risk.train import train

from .flow import retrain_flow
from .monitor import run_monitor
from .promote import auc_on


def cost_per_1000(model_dir: Path, X: pd.DataFrame, y: pd.Series, cost_fn: float, cost_fp: float) -> float:
    """Chi phí (đơn vị giả định) trên 1000 hồ sơ khi dùng mô hình + ngưỡng đã lưu của nó."""
    model = joblib.load(model_dir / "model.joblib")
    threshold = json.loads((model_dir / "metadata.json").read_text())["threshold"]
    reject = model.predict_proba(X)[:, 1] >= threshold
    fn = ((~reject) & (y == 1)).sum()  # cho vay người sẽ vỡ nợ
    fp = (reject & (y == 0)).sum()  # từ chối nhầm người tốt
    return round(1000 * (cost_fn * fn + cost_fp * fp) / len(y), 1)


log = logging.getLogger("pipelines.simulate")


def month_data(m: int, n: int, shock_month: int) -> pd.DataFrame:
    return make_synthetic(
        n,
        seed=1000 + m,
        covariate_shift=0.08 * m,
        concept_shift=0.8 if m >= shock_month else 0.0,
    )


def simulate(months: int = 12, n: int = 6000, shock_month: int = 6, out_dir: Path | None = None) -> pd.DataFrame:
    out_dir = out_dir or ROOT / "reports" / "sim"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    static_dir, auto_dir = out_dir / "static_model", out_dir / "auto_model"
    base = Settings()

    # Tháng 0: huấn luyện mô hình ban đầu, dùng chung cho cả hai cách vận hành
    m0 = month_data(0, n * 2, shock_month)
    train(m0, Settings(model_dir=static_dir, reports_dir=out_dir / "r0"), data_source="month-0", use_mlflow=False)
    shutil.copytree(static_dir, auto_dir)
    static_model = joblib.load(static_dir / "model.joblib")

    rows, previous = [], None
    for m in range(1, months + 1):
        cur = month_data(m, n, shock_month)
        X, y = cur[RAW_COLS], cur[TARGET]

        # Đo hiệu năng tháng m TRƯỚC khi dùng dữ liệu tháng m để retrain (công bằng với cả hai cách)
        auto_model = joblib.load(auto_dir / "model.joblib")
        drift = run_monitor(cur, auto_dir)
        row = {
            "month": m,
            "default_rate": round(float(y.mean()), 4),
            "max_feature_psi": float(drift.table["psi"].max()),
            "score_psi": drift.score_psi,
            "auc_static": round(auc_on(static_model, X, y), 4),
            "auc_auto": round(auc_on(auto_model, X, y), 4),
            "cost_static": cost_per_1000(static_dir, X, y, base.cost_fn, base.cost_fp),
            "cost_auto": cost_per_1000(auto_dir, X, y, base.cost_fn, base.cost_fp),
        }

        # Cuối tháng: nhãn của tháng m đã có (giả định đơn giản), chạy luồng retrain trên 2 tháng gần nhất
        recent = pd.concat([previous, cur]) if previous is not None else cur
        result = retrain_flow(recent, serving_dir=auto_dir, work_dir=out_dir / "candidates", use_mlflow=False)
        row["retrained"] = result.retrained
        row["promoted"] = bool(result.decision and result.decision["promote"])
        rows.append(row)
        previous = cur
        log.info("Tháng %d: %s", m, row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "simulation.csv", index=False)
    write_markdown(df, out_dir / "simulation.md", shock_month)
    plot(df, out_dir / "simulation.png", shock_month)
    shutil.rmtree(out_dir / "candidates", ignore_errors=True)
    return df


def write_markdown(df: pd.DataFrame, path: Path, shock_month: int) -> None:
    lines = [
        "# Giả lập 12 tháng: để nguyên mô hình và tự động retrain",
        "",
        f"Dữ liệu giả lập. Covariate shift tăng dần mỗi tháng, cú sốc concept shift từ tháng {shock_month}.",
        "",
        "| Tháng | Tỉ lệ vỡ nợ | PSI max | AUC để nguyên | AUC tự retrain | Chi phí để nguyên | Chi phí tự retrain "
        "| Retrain | Thay mô hình |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['month']} | {r['default_rate']:.3f} | {r['max_feature_psi']:.3f} "
            f"| {r['auc_static']:.4f} | {r['auc_auto']:.4f} | {r['cost_static']:.0f} | {r['cost_auto']:.0f} "
            f"| {'có' if r['retrained'] else ''} "
            f"| {'có' if r['promoted'] else ''} |"
        )
    lines += [
        "",
        f"AUC trung bình: để nguyên **{df['auc_static'].mean():.4f}**, tự retrain **{df['auc_auto'].mean():.4f}**.",
        f"Chi phí trung bình / 1000 hồ sơ: để nguyên **{df['cost_static'].mean():.0f}**, "
        f"tự retrain **{df['cost_auto'].mean():.0f}**.",
        "",
        "Chi phí là đơn vị giả định (cho vay người vỡ nợ = 10, từ chối nhầm người tốt = 1), chỉ để so sánh.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot(df: pd.DataFrame, path: Path, shock_month: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("Chưa cài matplotlib, bỏ qua biểu đồ.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, col, ylabel in [(axes[0], "auc", "ROC-AUC"), (axes[1], "cost", "Chi phí / 1000 hồ sơ")]:
        ax.plot(df["month"], df[f"{col}_static"], marker="o", label="Để nguyên")
        ax.plot(df["month"], df[f"{col}_auto"], marker="o", label="Tự retrain")
        for m in df.loc[df["promoted"], "month"]:
            ax.axvline(m + 0.5, color="grey", alpha=0.25)  # mô hình mới có hiệu lực từ tháng sau
        ax.axvline(shock_month - 0.5, color="red", linestyle="--", label="Cú sốc kinh tế")
        ax.set_xlabel("Tháng")
        ax.set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle("Giả lập 12 tháng (dữ liệu giả lập)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Giả lập drift và retrain tự động")
    parser.add_argument("--months", type=int, default=12)
    parser.add_argument("--n", type=int, default=6000, help="Số hồ sơ mỗi tháng")
    parser.add_argument("--shock-month", type=int, default=6)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = simulate(args.months, args.n, args.shock_month)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
