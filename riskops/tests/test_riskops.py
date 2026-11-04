import numpy as np
import pandas as pd
import pytest
from prometheus_client import CollectorRegistry, generate_latest
from riskops.drift import StreamDrift, split_columns
from riskops.prom import DriftGauges


def _frame(n: int, seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": rng.lognormal(10 + shift, 1, n),
            "is_transfer": (rng.random(n) < 0.2 + 0.5 * (shift > 0)).astype(float),
        }
    )


def test_cot_co_hai_gia_tri_la_phan_loai():
    assert split_columns(_frame(200, 0)) == (["amount"], ["is_transfer"])


def test_stream_drift_chi_bao_khi_luong_thay_doi():
    ref = _frame(3000, 0)
    stable, shifted = StreamDrift(ref, window=1000, every=250), StreamDrift(ref, window=1000, every=250)
    out_stable = [stable.update(r) for r in _frame(1000, 1).to_dict("records")]
    for r in _frame(1000, 2, shift=1.0).to_dict("records"):
        shifted.update(r)
    # chỉ tính ở sự kiện thứ 500, 750, 1000 (đủ min_events=500 và tới lượt every=250)
    assert [i + 1 for i, s in enumerate(out_stable) if s is not None] == [500, 750, 1000]
    assert not stable.last.drifted
    assert shifted.last.drifted and shifted.last.n_alert == 2


def test_gauge_prometheus_co_nhan_model():
    reg = CollectorRegistry()
    gauges = DriftGauges("fraud", registry=reg)
    tracker = StreamDrift(_frame(2000, 0), window=500, every=500, ref_scores=np.linspace(0, 1, 2000))
    summary = None
    for r, s in zip(_frame(500, 3, shift=1.0).to_dict("records"), np.linspace(0.5, 1, 500)):
        summary = tracker.update(r, s) or summary
    gauges.publish(summary, len(tracker.rows))
    text = generate_latest(reg).decode()
    assert 'risk_feature_psi{feature="amount",model="fraud"}' in text
    assert 'risk_drift_alert{model="fraud"} 1.0' in text
    assert "risk_score_psi" in text
    # tạo lại trên cùng registry không lỗi trùng tên (API tạo lại app trong test)
    DriftGauges("credit", registry=reg)


def test_registry_ghi_dang_ky_va_tai_ve(tmp_path):
    pytest.importorskip("mlflow")
    from riskops import registry

    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    f = tmp_path / "model.txt"
    f.write_text("mô hình")
    v1 = registry.log_and_register(uri, "demo", "demo-model", [f], {"a": 1}, {"pr_auc": 0.5},
                                   artifact_root=tmp_path / "art")  # fmt: skip
    assert registry.version_of(uri, "demo-model", "challenger") == v1 == "1"
    assert registry.version_of(uri, "demo-model", "champion") is None
    registry.set_alias(uri, "demo-model", "champion", v1)
    out = registry.fetch(uri, "models:/demo-model@champion", tmp_path / "dl")
    assert (out / "model.txt").read_text() == "mô hình"
