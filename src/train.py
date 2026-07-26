import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# Se reemplazó root_mean_squared_error por mean_squared_error para compatibilidad retroactiva
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from mlflow.tracking import MlflowClient
import os

from src.config import (
    MLFLOW_TRACKING_URI, EXPERIMENT_NAME_M1, EXPERIMENT_NAME_M2,
    DATA_PROCESSED_DIR, M1_MIN_AUC, M2_MAX_RMSE
)
from src.utils import logger

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def get_registered_champion_metric(model_name: str, metric_name: str):
    """Obtiene la métrica registrada del actual modelo campeón en producción."""
    client = MlflowClient()
    try:
        latest_versions = client.get_latest_versions(model_name, stages=["Production", "None"])
        if not latest_versions:
            return None
        latest_run_id = latest_versions[0].run_id
        run = client.get_run(latest_run_id)
        return run.data.metrics.get(metric_name)
    except Exception:
        return None

def train_models():
    # ==========================================
    # ENTRENAMIENTO M1: PROPENSION
    # ==========================================
    logger.info("Entrenando Modelo 1 (Propensión)...")
    mlflow.set_experiment(EXPERIMENT_NAME_M1)
    
    train_m1 = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "train_m1.csv"))
    val_m1 = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "val_m1.csv"))

    # 1. Definimos las columnas a excluir para evitar el Data Leakage
    columnas_a_excluir_m1 = [
        'target_venta', 
        'target_copamiento', 
        'id_cliente', 
        'id',
        'codunicocli',       # ID único del cliente (¡Excluido para evitar memorización!)
        'copamiento',        # Soplón directo (¡Fuga de datos de M2!)
        'monto',             # Soplón (Fuga de datos)
        'monto_aceptado'     # Soplón (Fuga de datos)
    ]

    y_train_m1 = train_m1['target_venta']
    X_train_m1 = train_m1.drop(columns=columnas_a_excluir_m1, errors='ignore')
    
    y_val_m1 = val_m1['target_venta']
    X_val_m1 = val_m1.drop(columns=columnas_a_excluir_m1, errors='ignore')

    # 🔍 DIAGNÓSTICO M1: Columnas e información del target
    logger.info("====================================================")
    logger.info(f"🔍 COLUMNAS FINALES USADAS EN M1: {list(X_train_m1.columns)}")
    logger.info("📊 DISTRIBUCIÓN DE CLASES TARGET M1 (Venta):")
    logger.info(f"\n{y_val_m1.value_counts(normalize=True)}")
    logger.info("====================================================")
    
    with mlflow.start_run() as run_m1:
        model_m1 = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model_m1.fit(X_train_m1, y_train_m1)
        
        preds_prob = model_m1.predict_proba(X_val_m1)[:, 1]
        auc_score = roc_auc_score(y_val_m1, preds_prob)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 8)
        mlflow.log_metric("auc_roc", auc_score)
        
        mlflow.sklearn.log_model(
            sk_model=model_m1, 
            artifact_path="model",
            registered_model_name=None
        )

        champion_auc = get_registered_champion_metric("Propension_Modelo_Campeon", "auc_roc")
        
        if champion_auc is None or auc_score > champion_auc:
            if auc_score >= M1_MIN_AUC:
                mlflow.register_model(f"runs:/{run_m1.info.run_id}/model", "Propension_Modelo_Campeon")
                logger.info(f"¡Nuevo Campeón Registrado M1! AUC: {auc_score:.4f} (Anterior: {champion_auc})")
            else:
                logger.warning(f"AUC del modelo ({auc_score:.4f}) por debajo de la calidad mínima requerida ({M1_MIN_AUC}).")
        else:
            logger.info(f"El modelo candidato ({auc_score:.4f}) no superó al campeón actual ({champion_auc:.4f}).")

    # ==========================================
    # ENTRENAMIENTO M2: COPAMIENTO    
    # ==========================================
    logger.info("Entrenando Modelo 2 (Copamiento)...")
    mlflow.set_experiment(EXPERIMENT_NAME_M2)
    
    train_m2 = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "train_m2.csv"))
    val_m2 = pd.read_csv(os.path.join(DATA_PROCESSED_DIR, "val_m2.csv"))
    
    columnas_a_excluir_m2 = [
        'target_copamiento', 'target_venta', 'id_cliente', 'id',
        'monto_aceptado', 'copamiento'
    ]

    y_train_m2 = train_m2['target_copamiento']
    X_train_m2 = train_m2.drop(columns=columnas_a_excluir_m2, errors='ignore')
    
    y_val_m2 = val_m2['target_copamiento']
    X_val_m2 = val_m2.drop(columns=columnas_a_excluir_m2, errors='ignore')

    # 🔍 UBICACIÓN CLAVE - DIAGNÓSTICO M2: Imprime el resumen estadístico del target económico
    logger.info("====================================================")
    logger.info("📊 ESTADÍSTICAS DEL TARGET DE COPAMIENTO EN VALIDACIÓN:")
    logger.info(f"\n{y_val_m2.describe()}")
    logger.info("====================================================")

    with mlflow.start_run() as run_m2:
        model_m2 = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10, 
            criterion='squared_error', 
            random_state=42
        )
        
        y_train_m2_log = np.log1p(y_train_m2)
        model_m2.fit(X_train_m2, y_train_m2_log)
        
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("criterion", "squared_error_with_log1p")
        
        preds_m2_log = model_m2.predict(X_val_m2)
        preds_m2 = np.expm1(preds_m2_log)
        
        if 'monto_ofertado' in X_val_m2.columns:
            preds_m2 = np.minimum(preds_m2, X_val_m2['monto_ofertado'])
        
        rmse = np.sqrt(mean_squared_error(y_val_m2, preds_m2))
        mae = mean_absolute_error(y_val_m2, preds_m2)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        mlflow.sklearn.log_model(
            sk_model=model_m2, 
            artifact_path="model",
            registered_model_name=None
        )
        
        champion_rmse = get_registered_champion_metric("Copamiento_Modelo_Campeon", "rmse")
        
        if champion_rmse is None or rmse < champion_rmse:
            if rmse <= M2_MAX_RMSE:
                mlflow.register_model(f"runs:/{run_m2.info.run_id}/model", "Copamiento_Modelo_Campeon")
                logger.info(f"¡Nuevo Campeón Registrado M2! RMSE: {rmse:.2f} (Anterior: {champion_rmse})")
            else:
                logger.warning(f"RMSE ({rmse:.2f}) excede límites aceptables de tolerancia ({M2_MAX_RMSE}).")
        else:
            logger.info(f"El candidato M2 ({rmse:.2f}) no supera al campeón actual ({champion_rmse:.2f}).")


if __name__ == "__main__":
    train_models()  