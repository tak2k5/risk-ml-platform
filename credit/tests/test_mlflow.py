import pytest

from credit_risk.config import Settings
from credit_risk.data import make_synthetic
from credit_risk.train import train

mlflow = pytest.importorskip("mlflow")


def test_training_registers_challenger(tmp_path):
    s = Settings(
        model_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        mlflow_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        mlflow_artifacts=tmp_path / "mlartifacts",
    )
    train(make_synthetic(1500, seed=3), s, data_source="synthetic-test", use_mlflow=True)

    client = mlflow.MlflowClient(tracking_uri=s.mlflow_uri)
    mv = client.get_model_version_by_alias(s.registered_model, "challenger")
    assert str(mv.version) == "1"
    assert (tmp_path / "mlartifacts").exists()
