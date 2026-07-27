"""
PIPELINE 2 - INFERENCIA (serving / batch scoring)
-------------------------------------------------
Responsabilidad única: cargar el modelo campeón vigente y generar las
predicciones de los periodos de producción (p10-p12).

Corre de forma recurrente (@monthly). Al terminar, DISPARA el pipeline de
monitoreo para evaluar drift sobre lo recién inferido.
"""
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow')

from src.inference import run_inference

default_args = {
    'owner': 'mlops_group_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pipeline_inferencia',
    default_args=default_args,
    description='Pipeline de inferencia recurrente sobre periodos de producción',
    schedule_interval='@monthly',  # Se ejecuta en cada nuevo periodo
    catchup=False,
) as dag:

    task_inference = PythonOperator(
        task_id='inferencias_meses_produccion',
        python_callable=run_inference,
    )

    # Al terminar la inferencia, se dispara el pipeline de monitoreo
    task_trigger_monitoreo = TriggerDagRunOperator(
        task_id='disparar_monitoreo',
        trigger_dag_id='pipeline_monitoreo',
        reset_dag_run=True,
    )

    task_inference >> task_trigger_monitoreo
