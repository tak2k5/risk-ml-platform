"""Dữ liệu PaySim (giao dịch tiền di động tổng hợp, Kaggle) và bộ giả lập cùng định dạng để test.

PaySim chỉ có "step" (giờ thứ mấy, 1..743). Để phát lại như một luồng sự kiện, mỗi giao dịch được gán thời điểm
ts = step*3600 + vị trí trong giờ đó (chia đều theo thứ tự dòng). Cách gán cố định nên offline và online thấy cùng ts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["tx_id", "ts", "type", "amount", "orig", "dest", "old_bal_orig", "new_bal_orig", "old_bal_dest",
           "new_bal_dest", "is_fraud"]  # fmt: skip
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def assign_ts(step: pd.Series) -> np.ndarray:
    """Giây kể từ đầu dữ liệu: các giao dịch trong cùng một giờ được rải đều trong giờ đó, giữ thứ tự dòng."""
    rank = step.groupby(step).cumcount().to_numpy()
    size = step.map(step.value_counts()).to_numpy()
    return (step.to_numpy() * 3600 + (rank * 3600 // size)).astype(np.int64)


def load_paysim(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "tx_id": np.arange(len(df)),
            "ts": assign_ts(df["step"]),
            "type": df["type"],
            "amount": df["amount"],
            "orig": df["nameOrig"],
            "dest": df["nameDest"],
            "old_bal_orig": df["oldbalanceOrg"],
            "new_bal_orig": df["newbalanceOrig"],
            "old_bal_dest": df["oldbalanceDest"],
            "new_bal_dest": df["newbalanceDest"],
            "is_fraud": df["isFraud"].astype(int),
        }
    )  # bỏ isFlaggedFraud: đó là luật của hệ thống cũ, không phải thông tin có sẵn khi chấm
    return out.sort_values(["ts", "tx_id"], kind="stable").reset_index(drop=True)


def make_synthetic(hours: int = 24 * 10, per_hour: int = 200, n_accounts: int = 3000, n_mules: int = 30,
                   fraud_per_hour: float = 1.5, seed: int = 0) -> pd.DataFrame:  # fmt: skip
    """Giao dịch GIẢ LẬP theo định dạng PaySim. Gian lận: chuyển (TRANSFER) gần hết số dư của nạn nhân vào một tài
    khoản trung gian, tài khoản này nhận nhiều khoản trong vài giờ rồi rút tiền (CASH_OUT). Chỉ để test và chạy thử."""
    rng = np.random.default_rng(seed)
    balance = rng.lognormal(10, 1.2, n_accounts)
    accounts = np.array([f"C{i:07d}" for i in range(n_accounts)])
    merchants = np.array([f"M{i:06d}" for i in range(500)])
    mules = rng.choice(accounts, n_mules, replace=False)
    rows = []
    for h in range(hours):
        n = rng.poisson(per_hour)
        for _ in range(n):
            o = rng.integers(n_accounts)
            t = rng.choice(TYPES, p=[0.22, 0.35, 0.01, 0.34, 0.08])
            amt = float(min(rng.lognormal(8, 1.3), balance[o] if t != "CASH_IN" else 1e9))
            dest = merchants[rng.integers(500)] if t == "PAYMENT" else accounts[rng.integers(n_accounts)]
            old = balance[o]
            new = old + amt if t == "CASH_IN" else max(old - amt, 0)
            balance[o] = new
            od = 0.0 if dest.startswith("M") else float(rng.lognormal(9, 1.5))
            rows.append([h, t, amt, accounts[o], dest, old, new, od, od + amt if od else 0.0, 0])
        for _ in range(rng.poisson(fraud_per_hour)):
            v = rng.integers(n_accounts)
            mule = mules[rng.integers(n_mules)]
            amt = float(balance[v]) * rng.uniform(0.9, 1.0)
            old = balance[v]
            balance[v] = old - amt
            # đôi khi hệ thống ghi sai số dư bên nhận (như PaySim): số dư đích không đổi
            od = float(rng.lognormal(9, 1.5))
            rows.append([h, "TRANSFER", amt, accounts[v], mule, old, old - amt, od, od + amt * (rng.random() < 0.5), 1])
            rows.append([h, "CASH_OUT", amt, mule, accounts[rng.integers(n_accounts)], amt, 0.0, 0.0, 0.0, 1])
    df = pd.DataFrame(rows, columns=["step", "type", "amount", "orig", "dest", "old_bal_orig", "new_bal_orig",
                                     "old_bal_dest", "new_bal_dest", "is_fraud"])  # fmt: skip
    df = df.sample(frac=1, random_state=seed).sort_values("step", kind="stable").reset_index(drop=True)
    df.insert(0, "tx_id", np.arange(len(df)))
    df.insert(1, "ts", assign_ts(df["step"]))
    return df.drop(columns="step")[COLUMNS]
