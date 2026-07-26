import pandas as pd
import mlflow.sklearn
import os
from src.config import DATA_RAW_DIR, DATA_PREDICTIONS_DIR, INFERENCE_PERIODS, MLFLOW_TRACKING_URI
from src.utils import logger

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def run_inference():
    """
    Busca los archivos mensuales correspondientes a inferencia (p10, p11, p12)
    y ejecuta de forma encadenada el Modelo 1 y Modelo 2.
    """
    logger.info("Iniciando Pipeline de Inferencia...")
    
    try:
        model_m1 = mlflow.sklearn.load_model("models:/Propension_Modelo_Campeon/latest")
        model_m2 = mlflow.sklearn.load_model("models:/Copamiento_Modelo_Campeon/latest")
    except Exception as e:
        logger.error(f"Error cargando modelos desde el registro: {e}")
        return
        
    for period_file in INFERENCE_PERIODS:
        filepath = os.path.join(DATA_RAW_DIR, period_file)
        if not os.path.exists(filepath):
            logger.warning(f"Periodo de producción {period_file} aún no disponible. Omitiendo.")
            continue
            
        logger.info(f"Realizando inferencia mensual para: {period_file}")
        df = pd.read_csv(filepath)
        
        # 🌟 SOLUCIÓN: Usar las características exactas del entrenamiento 🌟
        try:
            # Seleccionamos únicamente las columnas que espera el Modelo 1
            features_m1 = model_m1.feature_names_in_
            X_infer_m1 = df[features_m1].copy()
            
            # Seleccionamos únicamente las columnas que espera el Modelo 2
            features_m2 = model_m2.feature_names_in_
            X_infer_m2 = df[features_m2].copy()
        except AttributeError:
            # Fallback en caso de que los modelos no tengan guardado 'feature_names_in_'
            logger.warning("No se detectó 'feature_names_in_'. Aplicando filtrado manual de columnas.")
            columnas_a_excluir = [
                'target_venta', 'target_copamiento', 'id_cliente', 
                'key_value', 'partition', 'ent_1erlntcrallsfm01', 'grp_campecs06m'
            ]
            features_manuales = [c for c in df.columns if c not in columnas_a_excluir]
            X_infer_m1 = df[features_manuales].copy()
            X_infer_m2 = df[features_manuales].copy()
        
        # Imputación rápida para inferencia
        X_infer_m1 = X_infer_m1.fillna(X_infer_m1.median(numeric_only=True))
        X_infer_m2 = X_infer_m2.fillna(X_infer_m2.median(numeric_only=True))
        
        # Predicción M1 (Propensión)
        df['prob_venta'] = model_m1.predict_proba(X_infer_m1)[:, 1]
        df['pred_venta'] = model_m1.predict(X_infer_m1)
        
        # Predicción M2 (Copamiento: Solo si pred_venta es 1, de lo contrario monto es 0)
        df['pred_copamiento'] = 0.0
        aptos_idx = df[df['pred_venta'] == 1].index
        
        if len(aptos_idx) > 0:
            preds_copa = model_m2.predict(X_infer_m2.loc[aptos_idx])
            df.loc[aptos_idx, 'pred_copamiento'] = preds_copa
            
        os.makedirs(DATA_PREDICTIONS_DIR, exist_ok=True)
        output_file = os.path.join(DATA_PREDICTIONS_DIR, f"predictions_{period_file}")
        df.to_csv(output_file, index=False)
        logger.info(f"Predicciones para {period_file} guardadas en {output_file}")

if __name__ == "__main__":
    run_inference() 