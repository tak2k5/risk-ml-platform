"""Drift trên luồng (riskops.StreamDrift) và MLflow dùng chung (train --mlflow, promote)."""

import pandas as pd
import pytest

from fraudrt.consumer import DriftMonitor
from fraudrt.data import make_synthetic
from fraudrt.features import MemoryWindows
from fraudrt.promote import decide
from fraudrt.scorer import Scorer
from fraudrt.train import run


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("fraud")
    df = make_synthetic(hours=24 * 5, per_hour=120, seed=3)
    res = run(df, root / "models", root / "reports", "test")
    return root, df, res


def _replay(models, txs: pd.DataFrame) -> DriftMonitor:
    scorer = Scorer(MemoryWindows(), models)
    mon = DriftMonitor(models, window=1000, every=500, publish=False)
    for tx in txs.to_dict("records"):
        r = scorer.score(tx)
        mon.update(r["features"], r["score"])
    return mon


def test_train_luu_mau_tham_chieu_cho_drift(trained):
    root, _, _ = trained
    ref = pd.read_parquet(root / "models" / "reference.parquet")
    assert "score" in ref.columns and "amount" in ref.columns and len(ref) > 1000


def test_luong_on_dinh_khong_bao_drift_luong_lech_thi_bao(trained):
    root, df, _ = trained
    # phát lại 2 ngày giao dịch mới cùng phân phối (giả lập seed khác). Cửa sổ 1000 giao dịch cuối đã có đủ 24 giờ
    # lịch sử trong kho cửa sổ, như consumer chạy lâu với Redis: không drift
    stable = _replay(root / "models", make_synthetic(hours=48, per_hour=120, seed=9))
    assert stable.tracker.last is not None and not stable.tracker.last.drifted
    # luồng đổi hẳn: số tiền gấp 20 lần, chỉ còn TRANSFER (vd. một chiến dịch lừa đảo mới)
    shifted_df = make_synthetic(hours=48, per_hour=120, seed=9).assign(type="TRANSFER")
    shifted_df["amount"] *= 20
    shifted = _replay(root / "models", shifted_df)
    last = shifted.tracker.last
    assert last.drifted
    assert {"amount", "type_TRANSFER"} <= set(last.table.query("status == 'alert'").feature)


def test_quy_tac_len_champion():
    assert decide(None, 0.3).promote
    assert decide(0.30, 0.32).promote
    assert not decide(0.30, 0.302).promote


def test_mlflow_dung_chung_dang_ky_va_len_champion(trained, monkeypatch):
    pytest.importorskip("mlflow")
    from riskops.registry import version_of

    from fraudrt.promote import promote
    from fraudrt.train import log_to_mlflow

    root, _, res = trained
    uri = f"sqlite:///{root / 'mlflow.db'}"
    holdout = pd.read_parquet(root / "models" / "test_features.parquet")
    assert log_to_mlflow(res, root / "models", "test", uri) == "1"
    assert promote(uri, holdout).promote  # chưa có champion
    assert version_of(uri, "fraud-lgbm", "champion") == "1"
    # train lại y hệt: PR-AUC không tăng đủ -> giữ champion cũ
    assert log_to_mlflow(res, root / "models", "test", uri) == "2"
    d = promote(uri, holdout)
    assert not d.promote and d.champion_pr_auc == pytest.approx(d.challenger_pr_auc)
    assert version_of(uri, "fraud-lgbm", "champion") == "1"
