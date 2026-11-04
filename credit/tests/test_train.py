import json

import joblib


def test_training_writes_artifacts(trained):
    s, meta = trained
    assert s.model_path.exists()
    assert s.metadata_path.exists()
    assert (s.reports_dir / "metrics.md").exists()
    saved = json.loads(s.metadata_path.read_text())
    assert saved["model_version"] == meta["model_version"]


def test_selected_model_beats_random_and_threshold_in_range(trained):
    _, meta = trained
    assert meta["selected_model"] in {"logreg", "lgbm"}
    assert meta["test_metrics"][meta["selected_model"]]["roc_auc"] > 0.65
    assert 0 < meta["threshold"] < 1


def test_saved_pipeline_predicts_probabilities(trained, synthetic_df):
    s, _ = trained
    pipe = joblib.load(s.model_path)
    probs = pipe.predict_proba(synthetic_df.drop(columns=["TARGET"]).head(5))[:, 1]
    assert ((probs >= 0) & (probs <= 1)).all()
