import pandas as pd
import numpy as np
import os
from scipy.stats import ks_2samp
from src.config import (
    DATA_RAW_DIR, DATA_PROCESSED_DIR, MONITORING_DASHBOARDS_DIR,
    MONITORING_ALERTS_DIR, INFERENCE_PERIODS, DRIFT_THRESHOLD
)
from src.utils import logger

def check_drift_and_alerts():
    """
    Compara las distribuciones de las características de producción (p10-p12)
    contra los datos históricos (train_m1) usando el test Kolmogorov-Smirnov.
    """
    logger.info("Iniciando análisis de Data Drift nativo...")
    
    # 1. Crear directorios de salida si no existen
    os.makedirs(MONITORING_DASHBOARDS_DIR, exist_ok=True)
    os.makedirs(MONITORING_ALERTS_DIR, exist_ok=True)
    
    # 2. Cargar datos de referencia (Entrenamiento histórico)
    train_path = os.path.join(DATA_PROCESSED_DIR, "train_m1.csv")
    if not os.path.exists(train_path):
        logger.error(f"No se encontró el set de entrenamiento en {train_path}. Abortando monitoreo.")
        return
    
    df_ref = pd.read_csv(train_path)
    # Excluir la columna target si existe
    features = [col for col in df_ref.columns if col != 'target_venta']
    
    # 3. Cargar datos de producción actuales (p12_extrac.csv por ejemplo)
    prod_file = os.path.join(DATA_RAW_DIR, INFERENCE_PERIODS[-1]) # p12_extrac.csv
    if not os.path.exists(prod_file):
        logger.error(f"No se encontró el archivo de producción en {prod_file}. Abortando monitoreo.")
        return
        
    df_prod = pd.read_csv(prod_file)
    
    # 4. Calcular Kolmogorov-Smirnov para cada variable común
    drift_results = []
    drift_detected_count = 0
    
    for col in features:
        if col in df_prod.columns:
            # Eliminar nulos para el test estadístico
            ref_data = df_ref[col].dropna()
            prod_data = df_prod[col].dropna()
            
            if len(ref_data) > 0 and len(prod_data) > 0:
                stat, p_value = ks_2samp(ref_data, prod_data)
                # Si el p-value es muy pequeño (< 0.05), las distribuciones son estadísticamente distintas
                is_drift = p_value < 0.05
                if is_drift:
                    drift_detected_count += 1
                
                drift_results.append({
                    "Variable": col,
                    "KS Statistic": f"{stat:.4f}",
                    "p-value": f"{p_value:.4f}",
                    "Drift Detectado": "SÍ" if is_drift else "NO"
                })
    
    df_drift_report = pd.DataFrame(drift_results)
    
    # 5. Generar un reporte HTML visualmente espectacular y profesional
    html_report_path = os.path.join(MONITORING_DASHBOARDS_DIR, "drift_report.html")
    
    html_content = f"""
    <html>
    <head>
        <title>Reporte de Data Drift - MLOps Pipeline</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #f4f6f9; color: #333; }}
            h1 {{ color: #1e293b; border-bottom: 2px solid #3b82f6; padding-bottom: 10px; }}
            .summary {{ background-color: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: #fff; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-radius: 8px; overflow: hidden; }}
            th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #e2e8f0; }}
            th {{ background-color: #3b82f6; color: white; }}
            tr:hover {{ background-color: #f8fafc; }}
            .drift-yes {{ color: #ef4444; font-weight: bold; }}
            .drift-no {{ color: #10b981; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>📊 Reporte de Estabilidad y Data Drift (Kolmogorov-Smirnov)</h1>
        <div class="summary">
            <h3>Resumen del Pipeline de Monitoreo:</h3>
            <p><b>Set de Referencia:</b> Entrenamiento Histórico ({len(df_ref)} registros)</p>
            <p><b>Set de Producción Analizado:</b> {INFERENCE_PERIODS[-1]} ({len(df_prod)} registros)</p>
            <p><b>Porcentaje de Variables con Drift:</b> { (drift_detected_count / len(features)) * 100:.2f}%</p>
        </div>
        
        <table>
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Estadístico KS</th>
                    <th>p-value</th>
                    <th>¿Presenta Drift?</th>
                </tr>
            </thead>
            <tbody>
    """
    
    for row in drift_results:
        drift_class = "drift-yes" if row["Drift Detectado"] == "SÍ" else "drift-no"
        html_content += f"""
                <tr>
                    <td>{row['Variable']}</td>
                    <td>{row['KS Statistic']}</td>
                    <td>{row['p-value']}</td>
                    <td class="{drift_class}">{row['Drift Detectado']}</td>
                </tr>
        """
        
    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """
    
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    # 6. Guardar Alerta de Re-entrenamiento si más del 30% de variables tienen drift
    drift_ratio = drift_detected_count / len(features)
    if drift_ratio >= DRIFT_THRESHOLD:
        alert_msg = f"ALERTA: Se detectó Data Drift en el {drift_ratio*100:.2f}% de las variables. Se sugiere re-entrenamiento."
        alert_path = os.path.join(MONITORING_ALERTS_DIR, "drift_alert.txt")
        with open(alert_path, "w") as f:
            f.write(alert_msg)
        logger.warning(alert_msg)
    else:
        logger.info("Estabilidad de datos óptima. Sin alertas de drift activadas.")

if __name__ == "__main__":
    check_drift_and_alerts()  