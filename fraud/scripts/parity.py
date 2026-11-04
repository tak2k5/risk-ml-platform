"""Kiểm tra lệch đặc trưng offline / online (training-serving skew).

    python scripts/parity.py --online reports/online_features.jsonl --data data/paysim.csv
Tính lại đặc trưng offline bằng cách phát lại lịch sử (như lúc huấn luyện), so với đặc trưng consumer đã dùng cho
cùng tx_id. Lệch lớn hơn 1e-6 ở bất kỳ đặc trưng nào là lỗi.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fraudrt.data import load_paysim, make_synthetic  # noqa: E402
from fraudrt.features import FEATURES, offline_features  # noqa: E402


def compare(online: pd.DataFrame, offline: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    common = online.index.intersection(offline.index)
    diff = (online.loc[common, FEATURES] - offline.loc[common, FEATURES]).abs()
    return pd.DataFrame({"lệch lớn nhất": diff.max(), "số giao dịch lệch": (diff > tol).sum()}).assign(n=len(common))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", required=True)
    ap.add_argument("--data", default="data/paysim.csv")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start", type=float, default=0.85, help="producer bắt đầu từ đâu (khớp với --start của producer)")
    a = ap.parse_args()
    online = pd.read_json(a.online, lines=True).set_index("tx_id")
    df = make_synthetic() if a.synthetic else load_paysim(a.data)
    # Redis online chỉ thấy các giao dịch từ lúc producer bắt đầu: offline cũng phát lại đúng phần đó
    offline = offline_features(df.iloc[int(len(df) * a.start) :])
    res = compare(online, offline)
    print(res.to_markdown())
    bad = int(res["số giao dịch lệch"].sum())
    print(f"\n{res['n'].iloc[0]} giao dịch so sánh; {'KHÔNG lệch' if bad == 0 else f'{bad} giá trị lệch'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
