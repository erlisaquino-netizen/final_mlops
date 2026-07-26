import pandas as pd
import numpy as np
import mlflow.sklearn
import os

# Importamos directo desde tu configuración en src
from src.config import MLFLOW_TRACKING_URI
from src.utils import logger
 
# Configuramos MLflow usando la constante oficial de tu proyecto
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def probar_modelos_localmente():
    logger.info("🚀 Cargando modelos campeones desde el registro de MLflow...")
    try:
        # Cargamos Modelo 1 (Clasificación) y Modelo 2 (Regresión)
        model_m1 = mlflow.sklearn.load_model("models:/Propension_Modelo_Campeon/latest")
        model_m2 = mlflow.sklearn.load_model("models:/Copamiento_Modelo_Campeon/latest")
        logger.info("✅ ¡Modelos cargados exitosamente!")
    except Exception as e:
        logger.error(f"❌ Error al cargar los modelos. ¿Están registrados en MLflow?: {e}")
        return

    # Creación de datos de prueba simulados basados en las columnas del modelo M1
    logger.info("🔍 Creando datos de prueba simulados...")
    try:
        features_m1 = model_m1.feature_names_in_
        df_prueba = pd.DataFrame(np.random.randn(3, len(features_m1)), columns=features_m1)
    except AttributeError:
        logger.warning("El modelo no tiene 'feature_names_in_'. Usando datos de ejemplo genéricos.")
        df_prueba = pd.DataFrame({
            'edad': [35, 45, 28],
            'ingresos': [50000, 120000, 25000],
            'monto_ofertado': [10000, 20000, 5000]
        })

    X_m1 = df_prueba[model_m1.feature_names_in_] if hasattr(model_m1, 'feature_names_in_') else df_prueba
    X_m1 = X_m1.fillna(0)

    # ==========================================
    # INFERENCIA ENCADENADA (M1 -> M2)
    # ==========================================
    logger.info("--- Ejecutando Modelo 1 (Propensión) ---")
    df_prueba['prob_venta'] = model_m1.predict_proba(X_m1)[:, 1]
    df_prueba['pred_venta'] = model_m1.predict(X_m1)
    
    logger.info("--- Ejecutando Modelo 2 (Copamiento condicional) ---")
    df_prueba['pred_copamiento'] = 0.0 
    
    clientes_aptos_idx = df_prueba[df_prueba['pred_venta'] == 1].index
    
    if len(clientes_aptos_idx) > 0:
        logger.info(f"Se encontraron {len(clientes_aptos_idx)} cliente(s) apto(s).")
        X_m2 = df_prueba.loc[clientes_aptos_idx]
        if hasattr(model_m2, 'feature_names_in_'):
            X_m2 = X_m2.reindex(columns=model_m2.feature_names_in_, fill_value=0)
            
        preds_log = model_m2.predict(X_m2)
        monto_real_estimado = np.expm1(preds_log) # Revertir log1p
        df_prueba.loc[clientes_aptos_idx, 'pred_copamiento'] = monto_real_estimado
    else:
        logger.info("Ningún cliente fue clasificado como apto (pred_venta == 1) en esta prueba.")

    print("\n📊 Resultados Finales de la Inferencia:")
    print(df_prueba[['pred_venta', 'prob_venta', 'pred_copamiento']])

if __name__ == "__main__":
    probar_modelos_localmente()