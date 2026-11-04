"""Gửi N hồ sơ vay giả lập tới API tín dụng (theo lô), để có lưu lượng cho Grafana và đo drift.

python scripts/send_credit.py --n 600                 # cùng phân phối với lúc train
python scripts/send_credit.py --n 600 --shift 2.0     # "tháng mới" lệch mạnh -> risk_drift_alert{model="credit"} = 1
"""

from __future__ import annotations

import argparse

import httpx
from credit_risk.config import RAW_COLS
from credit_risk.data import make_synthetic


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--shift", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=100)
    a = ap.parse_args(argv)
    df = make_synthetic(a.n, seed=123, covariate_shift=a.shift)[RAW_COLS]
    df = df.astype(object).where(df.notna(), None)
    rows = df.to_dict("records")
    decisions: dict[str, int] = {}
    with httpx.Client(base_url=a.url, timeout=60) as c:
        for i in range(0, len(rows), a.batch):
            r = c.post("/predict/batch", json={"applications": rows[i : i + a.batch]})
            r.raise_for_status()
            for p in r.json()["predictions"]:
                decisions[p["decision"]] = decisions.get(p["decision"], 0) + 1
    print(f"Đã gửi {len(rows)} hồ sơ (shift={a.shift}): {decisions}")


if __name__ == "__main__":
    main()
