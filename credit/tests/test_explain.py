import pytest

from credit_risk.data import make_synthetic, split_xy
from credit_risk.explain import Explainer
from credit_risk.pipeline import build_pipeline


@pytest.mark.parametrize("kind", ["logreg", "lgbm"])
def test_top_reasons_only_push_risk_up(kind):
    X, y = split_xy(make_synthetic(1500, seed=11))
    pipe = build_pipeline(kind).fit(X, y)
    reasons = Explainer(pipe).top_reasons(X.head(5), k=3)

    assert len(reasons) == 5
    for rs in reasons:
        assert len(rs) <= 3
        # Lý do từ chối phải là yếu tố làm TĂNG rủi ro, xếp từ mạnh đến yếu
        contribs = [r.contribution for r in rs]
        assert all(c > 0 for c in contribs)
        assert contribs == sorted(contribs, reverse=True)


def test_readable_labels():
    from credit_risk.explain import readable

    assert readable("OCCUPATION_TYPE_Laborers") == "Nghề nghiệp = Laborers"
    assert readable("EXT_SOURCE_MEAN").startswith("Điểm tín dụng")
    assert readable("SOMETHING_ELSE") == "SOMETHING_ELSE"
