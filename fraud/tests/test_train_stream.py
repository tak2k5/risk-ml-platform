from fraudrt.alerts import LogAlerter, alert_text
from fraudrt.data import make_synthetic
from fraudrt.features import MemoryWindows
from fraudrt.scorer import Scorer
from fraudrt.train import run, time_split


def test_chia_theo_thoi_gian():
    df = make_synthetic(hours=40, per_hour=50)
    tr, va, te = time_split(df)
    assert tr.ts.max() < va.ts.min() and va.ts.max() < te.ts.min()


def test_huan_luyen_thang_doan_ngau_nhien_va_cham_tren_luong(tmp_path):
    df = make_synthetic(hours=24 * 5, per_hour=120, seed=3)
    res = run(df, tmp_path / "models", tmp_path / "reports", "test")
    assert res["full"]["PR-AUC"] > 5 * res["baseline"]
    assert (tmp_path / "reports" / "train.md").exists()
    # chấm trên luồng bằng mô hình đã lưu: phát lại phần cuối
    scorer = Scorer(MemoryWindows(), tmp_path / "models")
    out = [scorer.score(tx) for tx in df.tail(500).to_dict("records")]
    assert all(0 <= r["score"] <= 1 for r in out) and max(r["score_ms"] for r in out) < 1000


def test_canh_bao():
    a = LogAlerter()
    tx = {"tx_id": 5, "type": "TRANSFER", "amount": 1500000.0, "orig": "C1", "dest": "C2"}
    a(tx, 0.97)
    assert a.sent == [alert_text(tx, 0.97)] and "1,500,000" in a.sent[0]
