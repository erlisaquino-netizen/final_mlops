"""import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, TRAIN_PERIODS
from src.utils import logger, send_alert

def clean_and_prepare_data():
 
    logger.info("Iniciando Preprocesamiento y Análisis de Anomalías Temporales...")
    
    monthly_dfs = {}
    sales_rates = []
    
    # 1. Cargar datos e identificar la tasa de conversión histórica de cada periodo
    for period in TRAIN_PERIODS:
        filepath = os.path.join(DATA_RAW_DIR, period)
        if not os.path.exists(filepath):
            logger.warning(f"Falta el archivo {period} en raw_data, omitiendo.")
            continue
            
        df_temp = pd.read_csv(filepath)
        
        # Guardar para procesar luego si pasa los filtros
        monthly_dfs[period] = df_temp
        
        if 'target_venta' in df_temp.columns:
            conversion_rate = df_temp['target_venta'].mean()
            sales_rates.append(conversion_rate)
            logger.info(f"Periodo {period} - Tasa de conversión: {conversion_rate:.2%}")
            
    if not monthly_dfs:
        raise FileNotFoundError("No se encontraron archivos p1 a p9 en raw_data.")

    # [INNOVACIÓN] Detección de meses anómalos usando Desviación Absoluta de la Mediana (MAD)
    median_conv = np.median(sales_rates)
    mad = np.median([abs(x - median_conv) for x in sales_rates])
    # Umbral de tolerancia para descartar meses extremadamente atípicos
    threshold = 3.0 * mad if mad > 0 else 0.15 
    
    valid_dfs = []
    for period, df_temp in monthly_dfs.items():
        conv = df_temp['target_venta'].mean()
        if abs(conv - median_conv) > threshold:
            # Detectamos anomalía (ej. caída de sistemas o campaña de spam atípica)
            send_alert(
                message=f"Periodo {period} descartado automáticamente por anomalía de negocio (Conversión {conv:.2%} vs Mediana {median_conv:.2%})",
                alert_type="ANOMALY_EXCLUSION"
            )
        else:
            valid_dfs.append(df_temp)

    # Consolidar datos aprobados para entrenamiento
    df_consolidated = pd.concat(valid_dfs, ignore_index=True)
    logger.info(f"Dataset consolidado para entrenamiento: {df_consolidated.shape[0]} registros.")

    # Selección de variables
    exclude_cols = ['target_venta', 'target_copamiento', 'id_cliente']
    features = [c for c in df_consolidated.columns if c not in exclude_cols]
    
    # Imputación robusta
    for col in features:
        if df_consolidated[col].isnull().sum() > 0:
            if df_consolidated[col].dtype in [np.float64, np.int64]:
                df_consolidated[col] = df_consolidated[col].fillna(df_consolidated[col].median())
            else:
                df_consolidated[col] = df_consolidated[col].fillna(df_consolidated[col].mode()[0])

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    # --- SPLITS MODELO 1 ---
    X_m1 = df_consolidated[features]
    y_m1 = df_consolidated['target_venta']
    X_train_m1, X_val_m1, y_train_m1, y_val_m1 = train_test_split(
        X_m1, y_m1, test_size=0.2, random_state=42, stratify=y_m1
    )
    
    pd.concat([X_train_m1, y_train_m1], axis=1).to_csv(os.path.join(DATA_PROCESSED_DIR, "train_m1.csv"), index=False)
    pd.concat([X_val_m1, y_val_m1], axis=1).to_csv(os.path.join(DATA_PROCESSED_DIR, "val_m1.csv"), index=False)
    
    # --- SPLITS MODELO 2 ---
    # Modelo 2 evalúa únicamente clientes que aceptaron el producto (target_venta == 1)
    df_m2_base = df_consolidated[df_consolidated['target_venta'] == 1].copy()
    
    X_m2 = df_m2_base[features]
    y_m2 = df_m2_base['target_copamiento']
    X_train_m2, X_val_m2, y_train_m2, y_val_m2 = train_test_split(
        X_m2, y_m2, test_size=0.2, random_state=42
    )
    
    pd.concat([X_train_m2, y_train_m2], axis=1).to_csv(os.path.join(DATA_PROCESSED_DIR, "train_m2.csv"), index=False)
    pd.concat([X_val_m2, y_val_m2], axis=1).to_csv(os.path.join(DATA_PROCESSED_DIR, "val_m2.csv"), index=False)
    
    logger.info("Preprocesamiento y segmentación finalizada.")

if __name__ == "__main__":
    clean_and_prepare_data() 
"""




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import gc
from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, TRAIN_PERIODS
from src.utils import logger, send_alert

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimiza los tipos de datos en el DataFrame para ahorrar memoria RAM."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type in [np.float64, np.float32]:
            df[col] = df[col].astype(np.float32)
        elif col_type in [np.int64, np.int32]:
            # Identificar si es binario para optimizar a int8
            if df[col].nunique() <= 2:
                df[col] = df[col].astype(np.int8)
            else:
                df[col] = df[col].astype(np.int32)
    return df

def clean_and_prepare_data():
    """
    Carga de p1_extrac.csv a p9_extrac.csv. 
    Analiza y excluye automáticamente meses anómalos (por ejemplo, caídas drásticas de ventas).
    Genera los splits para entrenar Modelo 1 (Propensión) y Modelo 2 (Copamiento).
    """
    logger.info("Iniciando Preprocesamiento y Análisis de Anomalías Temporales...")
    
    monthly_dfs = {}
    sales_rates = []
    
    # 1. Cargar datos e identificar la tasa de conversión histórica de cada periodo
    for period in TRAIN_PERIODS:
        filepath = os.path.join(DATA_RAW_DIR, period)
        if not os.path.exists(filepath):
            logger.warning(f"Falta el archivo {period} en raw_data, omitiendo.")
            continue
            
        # Leer optimizando tipos desde el inicio
        df_temp = pd.read_csv(filepath)
        df_temp = optimize_dtypes(df_temp)
        
        monthly_dfs[period] = df_temp
        
        if 'target_venta' in df_temp.columns:
            conversion_rate = df_temp['target_venta'].mean()
            sales_rates.append(conversion_rate)
            logger.info(f"Periodo {period} - Tasa de conversión: {conversion_rate:.2%}")
            
    if not monthly_dfs:
        raise FileNotFoundError("No se encontraron archivos p1 a p9 en raw_data.")

    # [INNOVACIÓN] Detección de meses anómalos usando Desviación Absoluta de la Mediana (MAD)
    median_conv = np.median(sales_rates)
    mad = np.median([abs(x - median_conv) for x in sales_rates])
    threshold = 3.0 * mad if mad > 0 else 0.15 
    
    valid_dfs = []
    for period, df_temp in monthly_dfs.items():
        conv = df_temp['target_venta'].mean()
        if abs(conv - median_conv) > threshold:
            send_alert(
                message=f"Periodo {period} descartado automáticamente por anomalía de negocio (Conversión {conv:.2%} vs Mediana {median_conv:.2%})",
                alert_type="ANOMALY_EXCLUSION"
            )
        else:
            valid_dfs.append(df_temp)

    # Consolidar datos aprobados para entrenamiento
    df_consolidated = pd.concat(valid_dfs, ignore_index=True)
    
    # Liberar memoria de los dataframes intermedios inmediatamente
    del monthly_dfs
    del valid_dfs
    gc.collect()

    logger.info(f"Dataset consolidado antes de muestreo: {df_consolidated.shape[0]} registros.")

    # [OPTIMIZACIÓN DE MEMORIA] Estratificado por muestreo si el set de datos supera los 500k registros
    MAX_ROWS = 500000
    if df_consolidated.shape[0] > MAX_ROWS:
        logger.info(f"Aplicando muestreo estratificado al 15% para proteger la memoria RAM de Docker...")
        df_consolidated, _ = train_test_split(
            df_consolidated, 
            train_size=0.15, 
            random_state=42, 
            stratify=df_consolidated['target_venta']
        )
        df_consolidated = df_consolidated.reset_index(drop=True)
        logger.info(f"Nuevo tamaño de dataset optimizado para RAM: {df_consolidated.shape[0]} registros.")
        gc.collect()

    # Selección de variables
    #exclude_cols = ['target_venta', 'target_copamiento', 'id_cliente']
    #features = [c for c in df_consolidated.columns if c not in exclude_cols]
    

    # Selección de variables (excluyendo targets e IDs)
    exclude_cols = ['target_venta', 'target_copamiento', 'id_cliente', 'periodo', 'period']
    
    # [OPTIMIZACIÓN] Nos aseguramos de mantener ÚNICAMENTE las columnas que sean numéricas
    features = [
        c for c in df_consolidated.columns 
        if c not in exclude_cols and df_consolidated[c].dtype in [np.float32, np.float64, np.int32, np.int64, np.int8]
    ]
    
    logger.info(f"Variables seleccionadas para el entrenamiento ({len(features)}): {features}")


    
    # Imputación robusta optimizada
    for col in features:
        if df_consolidated[col].isnull().sum() > 0:
            if df_consolidated[col].dtype in [np.float32, np.float64, np.int32, np.int64, np.int8]:
                median_val = df_consolidated[col].median()
                df_consolidated[col] = df_consolidated[col].fillna(median_val)
            else:
                mode_val = df_consolidated[col].mode()[0]
                df_consolidated[col] = df_consolidated[col].fillna(mode_val)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    
    # --- SPLITS MODELO 1 ---
    X_m1 = df_consolidated[features]
    y_m1 = df_consolidated['target_venta']
    X_train_m1, X_val_m1, y_train_m1, y_val_m1 = train_test_split(
        X_m1, y_m1, test_size=0.2, random_state=42, stratify=y_m1
    )
    
    pd.concat([X_train_m1, y_train_m1], axis=1).to_csv(os.path.join(DATA_PROCESSED_DIR, "train_m1.csv"), index=False)
    pd.concat([X_val_m1, y_val_m1], axis=1).to_csv(os.path.join(DATA_PROCESSED_DIR, "val_m1.csv"), index=False)
    
    # Liberar memoria de M1
    del X_train_m1, X_val_m1, y_train_m1, y_val_m1
    gc.collect()
    
    # --- SPLITS MODELO 2 ---
    df_m2_base = df_consolidated[df_consolidated['target_venta'] == 1].copy()
    
    # Liberar el dataset consolidado gigante
    del df_consolidated
    gc.collect()
    
    X_m2 = df_m2_base[features]
    y_m2 = df_m2_base['target_copamiento']
    X_train_m2, X_val_m2, y_train_m2, y_val_m2 = train_test_split(
        X_m2, y_m2, test_size=0.2, random_state=42
    )
    
    pd.concat([X_train_m2, y_train_m2], axis=1).to_csv(os.path.join(DATA_PROCESSED_DIR, "train_m2.csv"), index=False)
    pd.concat([X_val_m2, y_val_m2], axis=1).to_csv(os.path.join(DATA_PROCESSED_DIR, "val_m2.csv"), index=False)
    
    logger.info("Preprocesamiento y segmentación finalizada.")

if __name__ == "__main__":
    clean_and_prepare_data()