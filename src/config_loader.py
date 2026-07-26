"""Carga centralizada del único archivo de configuración del proyecto."""
from pathlib import Path
import os
import yaml

DEFAULT_CONFIG = "/opt/airflow/config/pipeline_config.yml"


def load_config(path: str | None = None) -> dict:
    config_path = Path(path or os.getenv("PIPELINE_CONFIG", DEFAULT_CONFIG))
    if not config_path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuración: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)
