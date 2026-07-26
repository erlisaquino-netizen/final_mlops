"""Funciones pequeñas para registrar runs y versiones en MLflow."""
import os
from pathlib import Path
import mlflow
import mlflow.sklearn


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))


def log_candidate(experiment: str, run_name: str, model, params: dict, metrics: dict, artifacts: list[str] | None = None) -> str:
    configure_mlflow()
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.sklearn.log_model(model, artifact_path="model")
        for artifact in artifacts or []:
            if Path(artifact).exists():
                mlflow.log_artifact(artifact)
        return run.info.run_id


def register_selected_model(run_id: str, registry_name: str, alias: str = "champion") -> int:
    configure_mlflow()
    result = mlflow.register_model(f"runs:/{run_id}/model", registry_name)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(registry_name, alias, result.version)
    return int(result.version)
