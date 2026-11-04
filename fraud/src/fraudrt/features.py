"""Đặc trưng của một giao dịch, tính ĐÚNG THỜI ĐIỂM: chỉ dùng các giao dịch xảy ra trước nó.

Cùng một hàm `featurize` dùng cho cả huấn luyện (phát lại lịch sử qua MemoryWindows) và chấm điểm trực tuyến (qua
RedisWindows). Hai kho chỉ khác chỗ lưu, cùng logic, nên đặc trưng offline và online giống nhau (không có
training-serving skew); scripts/parity.py kiểm tra lại bằng số.

Đặc trưng cửa sổ (cho cả tài khoản gửi và tài khoản nhận):
- số giao dịch trong 1 giờ trước, tổng tiền trong 24 giờ trước, số giao dịch trong 24 giờ trước
"""

from __future__ import annotations

import collections
import math

from .data import TYPES

HOUR, DAY = 3600, 86400
WINDOW_FEATURES = [f"{side}_{name}" for side in ("orig", "dest") for name in ("cnt_1h", "cnt_24h", "sum_24h")]
ROW_FEATURES = [f"type_{t}" for t in TYPES] + [
    "amount", "log_amount", "amount_over_balance", "orig_balance_error", "dest_balance_error", "dest_old_zero",
    "orig_emptied", "hour_of_day",
]  # fmt: skip
FEATURES = ROW_FEATURES + WINDOW_FEATURES


class MemoryWindows:
    """Lịch sử giao dịch của mỗi tài khoản trong RAM (dùng khi huấn luyện và test)."""

    def __init__(self):
        self.events: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.seen: dict[str, set] = collections.defaultdict(set)

    def stats(self, account: str, ts: int) -> tuple[int, int, float]:
        q = self.events[account]
        while q and q[0][0] <= ts - DAY:  # bỏ giao dịch cũ hơn 24 giờ
            self.seen[account].discard(q[0][1])
            q.popleft()
        cnt_1h = sum(1 for t, _, _ in q if ts - HOUR < t < ts)
        in_day = [(t, a) for t, _, a in q if ts - DAY < t < ts]
        return cnt_1h, len(in_day), sum(a for _, a in in_day)

    def add(self, account: str, ts: int, tx_id: int, amount: float) -> None:
        if tx_id in self.seen[account]:  # nhận lại cùng giao dịch (consumer khởi động lại): không đếm hai lần
            return
        self.seen[account].add(tx_id)
        self.events[account].append((ts, tx_id, amount))


class RedisWindows:
    """Cùng logic, lưu trong Redis: mỗi tài khoản một sorted set, điểm = ts, phần tử = "tx_id:amount".

    ZADD với cùng phần tử là không đổi gì, nên xử lý lại một giao dịch không làm sai đặc trưng.
    """

    def __init__(self, client, prefix: str = "w:"):
        self.r, self.prefix = client, prefix

    def stats(self, account: str, ts: int) -> tuple[int, int, float]:
        key = self.prefix + account
        pipe = self.r.pipeline()
        pipe.zremrangebyscore(key, "-inf", ts - DAY)
        pipe.zcount(key, f"({ts - HOUR}", f"({ts}")
        pipe.zrangebyscore(key, f"({ts - DAY}", f"({ts}")
        _, cnt_1h, members = pipe.execute()
        amounts = [float(m.decode().split(":")[1] if isinstance(m, bytes) else m.split(":")[1]) for m in members]
        return int(cnt_1h), len(amounts), sum(amounts)

    def add(self, account: str, ts: int, tx_id: int, amount: float) -> None:
        key = self.prefix + account
        pipe = self.r.pipeline()
        pipe.zadd(key, {f"{tx_id}:{amount!r}": ts})
        pipe.expire(key, DAY + HOUR)  # tài khoản không hoạt động thì tự xóa
        pipe.execute()


def row_features(tx: dict) -> dict:
    amt, old, new = float(tx["amount"]), float(tx["old_bal_orig"]), float(tx["new_bal_orig"])
    od, nd = float(tx["old_bal_dest"]), float(tx["new_bal_dest"])
    sign = 1 if tx["type"] == "CASH_IN" else -1
    f = {f"type_{t}": float(tx["type"] == t) for t in TYPES}
    f.update(
        amount=amt,
        log_amount=math.log1p(amt),
        amount_over_balance=amt / old if old > 0 else -1.0,
        # số dư sau giao dịch khớp với số tiền không? Lệch là dấu hiệu bất thường trong PaySim
        orig_balance_error=old + sign * amt - new,
        dest_balance_error=od - sign * amt - nd if not str(tx["dest"]).startswith("M") else 0.0,
        dest_old_zero=float(od == 0),
        orig_emptied=float(old > 0 and new == 0),
        hour_of_day=float(int(tx["ts"]) // HOUR % 24),
    )
    return f


def featurize(store, tx: dict, update: bool = True) -> dict:
    """Đặc trưng của giao dịch tx từ trạng thái TRƯỚC nó, rồi (mặc định) ghi tx vào kho cửa sổ."""
    f = row_features(tx)
    ts = int(tx["ts"])
    for side in ("orig", "dest"):
        c1, c24, s24 = store.stats(str(tx[side]), ts)
        f[f"{side}_cnt_1h"], f[f"{side}_cnt_24h"], f[f"{side}_sum_24h"] = float(c1), float(c24), float(s24)
    if update:
        for side in ("orig", "dest"):
            store.add(str(tx[side]), ts, int(tx["tx_id"]), float(tx["amount"]))
    return f


def offline_features(df, store=None):
    """Phát lại toàn bộ lịch sử theo thứ tự thời gian -> bảng đặc trưng (cùng hàm với online)."""
    import pandas as pd

    store = store or MemoryWindows()
    rows = [featurize(store, tx) for tx in df.sort_values(["ts", "tx_id"], kind="stable").to_dict("records")]
    out = pd.DataFrame(rows, columns=FEATURES)
    out.index = df.sort_values(["ts", "tx_id"], kind="stable")["tx_id"].to_numpy()
    return out
