"""
PIPELINE 1 - ENTRENAMIENTO (offline)
------------------------------------
Responsabilidad única: preparar los datos históricos y entrenar/registrar los
modelos (comparación Campeón vs Retador en MLflow).

No tiene 'schedule' propio: se ejecuta de forma MANUAL para el modelo inicial y
es DISPARADO automáticamente por el pipeline de monitoreo cuando detecta drift.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow')

from src.preprocess import clean_and_prepare_data
from src.train import train_models

default_args = {
    'owner': 'mlops_group_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pipeline_entrenamiento',
    default_args=default_args,
    description='Pipeline de entrenamiento: preprocesamiento + entrenamiento y registro de modelos',
    schedule_interval=None,  # Solo manual o disparado por el pipeline de monitoreo
    catchup=False,
) as dag:

    task_preprocess = PythonOperator(
        task_id='preprocesar_datos_entrenamiento',
        python_callable=clean_and_prepare_data,
    )

    task_train = PythonOperator(
        task_id='entrenar_y_registrar_modelos',
        python_callable=train_models,
    )

    task_preprocess >> task_train
