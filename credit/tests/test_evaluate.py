import numpy as np

from credit_risk.evaluate import choose_threshold, compute_metrics


def test_perfect_model_metrics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p)
    assert m.roc_auc == 1.0
    assert m.ks == 1.0


def test_threshold_prefers_blocking_defaults_when_fn_is_expensive():
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=2000) < 0.1).astype(int)
    p = np.clip(0.1 + 0.3 * y + rng.normal(0, 0.1, size=2000), 0, 1)

    cheap_fn = choose_threshold(y, p, cost_fn=1, cost_fp=1)
    pricey_fn = choose_threshold(y, p, cost_fn=20, cost_fp=1)
    # FN càng đắt thì ngưỡng càng thấp (từ chối nhiều hơn) và chặn được nhiều người vỡ nợ hơn
    assert pricey_fn.threshold <= cheap_fn.threshold
    assert pricey_fn.recall_default >= cheap_fn.recall_default
