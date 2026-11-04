"""MLflow dùng chung: ghi lần huấn luyện, đăng ký phiên bản, đặt alias, tải mô hình về để phục vụ.

Cả hai mô hình nằm trong MỘT MLflow (docker compose: http://localhost:5000):

    credit-risk   (credit/src/credit_risk/train.py)   alias challenger sau khi train, champion sau khi duyệt
    fraud-lgbm    (fraud/src/fraudrt/train.py)        như trên

Dịch vụ phục vụ (API tín dụng, consumer gian lận) có thể nạp thẳng `models:/<tên>@champion` qua `fetch()`, nên đổi
mô hình chỉ là đổi alias trong registry, không phải build lại image hay chép file.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("riskops.registry")


def _mlflow(tracking_uri: str):
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    return mlflow


def log_and_register(
    tracking_uri: str,
    experiment: str,
    model_name: str,
    files: list[Path],
    params: dict | None = None,
    metrics: dict | None = None,
    run_name: str | None = None,
    alias: str = "challenger",
    artifact_root: Path | None = None,
) -> str:
    """Ghi một lần chạy (tham số, chỉ số, các file mô hình trong thư mục artifact `model/`), tạo phiên bản mới
    trong registry và gắn `alias`. Trả về số phiên bản.

    `artifact_root`: khi tracking là file/sqlite (không có server), nơi lưu artifact cho experiment mới.
    """
    mlflow = _mlflow(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        location = None
        if artifact_root is not None and not tracking_uri.startswith("http"):
            location = Path(artifact_root).resolve().as_uri()
        mlflow.create_experiment(experiment, artifact_location=location)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        for f in files:
            mlflow.log_artifact(str(f), artifact_path="model")
    client = mlflow.MlflowClient()
    try:
        client.create_registered_model(model_name)
    except Exception:  # đã tồn tại
        pass
    mv = client.create_model_version(model_name, source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id)
    client.set_registered_model_alias(model_name, alias, mv.version)
    log.info("Đã đăng ký %s phiên bản %s (alias %s)", model_name, mv.version, alias)
    return str(mv.version)


def version_of(tracking_uri: str, model_name: str, alias: str) -> str | None:
    """Số phiên bản đang mang `alias`, hoặc None nếu chưa có."""
    client = _mlflow(tracking_uri).MlflowClient()
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except Exception:
        return None


def set_alias(tracking_uri: str, model_name: str, alias: str, version: str) -> None:
    _mlflow(tracking_uri).MlflowClient().set_registered_model_alias(model_name, alias, version)
    log.info("%s: alias %s -> phiên bản %s", model_name, alias, version)


def fetch(tracking_uri: str, model_uri: str, dest: Path) -> Path:
    """Tải các file của một phiên bản (vd `models:/fraud-lgbm@champion`) về `dest`, trả về thư mục chứa file."""
    mlflow = _mlflow(tracking_uri)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    out = Path(mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=str(dest)))
    log.info("Đã tải %s về %s", model_uri, out)
    return out
