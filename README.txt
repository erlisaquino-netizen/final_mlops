========================================================================
             MANUAL DE USUARIO Y PRUEBA FUNCIONAL: PIPELINE MLOPS
========================================================================
 
DESCRIPCIÓN GENERAL:
Este sistema implementa una solución de producción MLOps integrada por Airflow (orquestación), MLflow (experimentación y registro de modelos), Scikit-Learn (modelado) y Evidently AI junto con scipy (monitoreo de drift).

PASOS DE EJECUCIÓN (Evaluación funcional):

1. LEVANTAR LA PLATAFORMA
------------------------------------------------------------------------
Abra una terminal en el directorio raíz "proyecto-mlops-final/" y ejecute:
$ docker compose up --build -d

Verifique que todos los contenedores levanten sin errores:
$ docker compose ps

Puertos habilitados:
- Airflow Webserver: http://localhost:8080 (user: admin, pass: admin)
- MLflow Tracking: http://localhost:5000

2. MODO TRAINING (Prueba de inicio rápido)
------------------------------------------------------------------------
- Coloque su dataset original del curso en: "data/raw_data/train_propension.csv".
- Acceda a Airflow (http://localhost:8080), active el DAG "pipeline_mlops_propension_copamiento" y ejecútelo manualmente de manera inicial.
- Esto disparará:
  1. Preprocesamiento (creando sets limpios en data/processed/).
  2. Entrenamiento de ambos modelos (M1 y M2) y su registro automático en MLflow.
- Acceda a MLflow (http://localhost:5000) para verificar los experimentos y artefactos guardados en "registry/".

3. MODO INFERENCE (Simulación de producción)
------------------------------------------------------------------------
- Coloque un nuevo conjunto de datos de clientes en "data/raw_data/nuevos_datos.csv".
- Ejecute el DAG de Airflow de nuevo. El sistema detectará el nuevo archivo, predecirá propensión (M1) y, si es positiva, estimará el monto ideal con M2.
- Verifique las predicciones generadas en: "data/predictions/predictions_nuevos_datos.csv".

4. MONITOREO DE DRIFT Y RE-ENTRENAMIENTO AUTOMÁTICO
------------------------------------------------------------------------
- Para simular Drift deliberado, añada el archivo "data/raw_data/nuevos_datos_drift.csv" (el cual debe contener distribuciones demográficas o de ingresos radicalmente distintas al de entrenamiento).
- Corra el DAG en Airflow. 
- La tarea 'monitorear_drift' detectará que la métrica de Drift supera el 10% tolerado (mediante KS-Test).
- Automáticamente se disparará:
  - Una alerta en formato JSON guardada en "monitoring/alerts/".
  - Un reporte web interactivo interactivo de Evidently en "monitoring/dashboards/evidently_drift_report.html".
  - El trigger de "ShortCircuit" de Airflow se mantendrá activo y disparará inmediatamente la tarea final 'disparar_reentrenamiento'.

PUNTOS DE INNOVACIÓN INCORPORADOS (Para sustentación de nota máxima):
- Filtro inteligente y exclusión automática de anomalías extremas antes de entrenar (limpieza proactiva).
- Comparador automático Campeón vs Retador: MLflow no registra un modelo nuevo a menos de que supere la métrica AUC del modelo ya existente.
- Sistema estructurado de alertas de Drift vía ficheros log listos para conectar con Webhooks de Slack/Teams.

--http://localhost:5000/
--http://localhost:8080/

--docker compose exec -u airflow airflow-scheduler python /opt/airflow/src/predict.py