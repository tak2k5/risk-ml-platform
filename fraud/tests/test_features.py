import pandas as pd
import pytest

from fraudrt.data import assign_ts, make_synthetic
from fraudrt.features import FEATURES, MemoryWindows, RedisWindows, featurize, offline_features


class FakeRedis:
    """Đủ các lệnh sorted set mà RedisWindows dùng, để test cùng logic Redis mà không cần máy chủ."""

    def __init__(self):
        self.z = {}

    def pipeline(self):
        return FakePipe(self)


class FakePipe:
    def __init__(self, r):
        self.r, self.ops = r, []

    def __getattr__(self, name):
        return lambda *a, **k: self.ops.append((name, a, k))

    @staticmethod
    def _bound(v, low):
        if v in ("-inf", "+inf"):
            return float(v), False
        s = str(v)
        return (float(s[1:]), True) if s.startswith("(") else (float(s), False)

    def _in(self, score, lo, hi):
        (a, ax), (b, bx) = self._bound(lo, True), self._bound(hi, False)
        return (score > a if ax else score >= a) and (score < b if bx else score <= b)

    def execute(self):
        out = []
        for name, a, _ in self.ops:
            z = self.r.z.setdefault(a[0], {})
            if name == "zadd":
                z.update(a[1])
                out.append(1)
            elif name == "zremrangebyscore":
                for m in [m for m, s in z.items() if self._in(s, a[1], a[2])]:
                    del z[m]
                out.append(0)
            elif name == "zcount":
                out.append(sum(self._in(s, a[1], a[2]) for s in z.values()))
            elif name == "zrangebyscore":
                out.append([m.encode() for m, s in sorted(z.items(), key=lambda kv: kv[1]) if self._in(s, a[1], a[2])])
            else:
                out.append(True)
        return out


def tx(i, ts, orig="A", dest="B", amount=100.0, type_="TRANSFER"):
    return {"tx_id": i, "ts": ts, "type": type_, "amount": amount, "orig": orig, "dest": dest,
            "old_bal_orig": 1000.0, "new_bal_orig": 900.0, "old_bal_dest": 0.0, "new_bal_dest": 0.0}  # fmt: skip


@pytest.mark.parametrize("store_cls", [MemoryWindows, lambda: RedisWindows(FakeRedis())])
def test_cua_so_chi_tinh_giao_dich_truoc(store_cls):
    s = store_cls()
    featurize(s, tx(1, 0, amount=10))
    featurize(s, tx(2, 1800, amount=20))
    f = featurize(s, tx(3, 3600, amount=40))  # 1 giờ sau giao dịch đầu: giao dịch đầu ra khỏi cửa sổ 1h
    assert f["orig_cnt_1h"] == 1 and f["orig_cnt_24h"] == 2 and f["orig_sum_24h"] == 30
    f = featurize(s, tx(4, 86400 + 1800, amount=1))  # 24h sau: chỉ còn giao dịch lúc 3600
    assert f["orig_cnt_24h"] == 1 and f["orig_sum_24h"] == 40


@pytest.mark.parametrize("store_cls", [MemoryWindows, lambda: RedisWindows(FakeRedis())])
def test_xu_ly_lai_khong_dem_trung(store_cls):
    s = store_cls()
    featurize(s, tx(1, 0))
    featurize(s, tx(1, 0))  # consumer khởi động lại, nhận lại giao dịch 1
    assert featurize(s, tx(2, 60))["orig_cnt_1h"] == 1


def test_giao_dich_hien_tai_khong_tu_tinh_vao_dac_trung_cua_no():
    f = featurize(MemoryWindows(), tx(1, 100))
    assert f["orig_cnt_1h"] == 0 and f["dest_sum_24h"] == 0


def test_offline_va_online_cho_cung_dac_trung():
    df = make_synthetic(hours=30, per_hour=60, seed=1)
    off = offline_features(df)
    on = offline_features(df, RedisWindows(FakeRedis()))
    pd.testing.assert_frame_equal(off, on)
    assert list(off.columns) == FEATURES


def test_khong_dung_thong_tin_tuong_lai():
    df = make_synthetic(hours=30, per_hour=60, seed=2)
    a = offline_features(df)
    cut = df.ts.quantile(0.5)
    b = offline_features(df[df.ts <= cut])  # bỏ toàn bộ tương lai
    pd.testing.assert_frame_equal(a.loc[b.index], b)


def test_gan_ts_trong_gio():
    ts = assign_ts(pd.Series([1, 1, 1, 2]))
    assert list(ts) == [3600, 4800, 6000, 7200]
