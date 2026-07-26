import os

# Ruta base dentro del contenedor de Airflow (/opt/airflow)
BASE_DIR = "/opt/airflow"

# Carpetas mapeadas directamente
DATA_RAW_DIR = os.path.join(BASE_DIR, "raw_data")       # Coincide con ./raw_data
REGISTRY_DIR = os.path.join(BASE_DIR, "registry")       # Coincide con ./registry

# Carpetas que se crearán automáticamente para procesados, predicciones y monitoreo
DATA_PROCESSED_DIR = os.path.join(DATA_RAW_DIR, "processed") 
DATA_PREDICTIONS_DIR = os.path.join(DATA_RAW_DIR, "predictions")
MONITORING_DASHBOARDS_DIR = os.path.join(DATA_RAW_DIR, "monitoring", "dashboards")
MONITORING_ALERTS_DIR = os.path.join(DATA_RAW_DIR, "monitoring", "alerts")

# Definición de periodos
TRAIN_PERIODS = [f"p{i}_extrac.csv" for i in range(1, 10)]
INFERENCE_PERIODS = [f"p{i}_extrac.csv" for i in range(10, 13)]

# MLflow config
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME_M1 = "Propension_Compra_M1"
EXPERIMENT_NAME_M2 = "Copamiento_M2"

# Umbrales
DRIFT_THRESHOLD = 0.15
M1_MIN_AUC = 0.65
M2_MAX_RMSE = 10000.0  