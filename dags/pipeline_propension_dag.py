from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow')

from src.preprocess import clean_and_prepare_data
from src.train import train_models
from src.inference import run_inference
from src.monitor import check_drift_and_alerts

default_args = {
    'owner': 'mlops_group_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pipeline_orquestado_extrac_mlops',
    default_args=default_args,
    description='Orquestación de MLOps de Propensión y Copamiento con datos temporales (p1-p12)',
    schedule_interval='@monthly', # Ajustado a la naturaleza mensual de los datos
    catchup=False,
) as dag:

    task_preprocess = PythonOperator(
        task_id='preprocesar_datos_entrenamiento',
        python_callable=clean_and_prepare_data,
    )

    task_train = PythonOperator(
        task_id='entrenar_y_comparar_modelos',
        python_callable=train_models,
    )

    task_inference = PythonOperator(
        task_id='inferencias_meses_produccion',
        python_callable=run_inference,
    )

    # Si retorna True, se activa el re-entrenamiento automático
    task_check_drift = ShortCircuitOperator(
        task_id='monitorear_y_evaluar_drift',
        python_callable=check_drift_and_alerts,
    )

    task_retrain = PythonOperator(
        task_id='disparar_reentrenamiento',
        python_callable=train_models,
    )

    task_preprocess >> task_train >> task_inference >> task_check_drift >> task_retrain  