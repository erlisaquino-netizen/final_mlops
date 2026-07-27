"""
PIPELINE 3 - MONITOREO (drift)
------------------------------
Responsabilidad única: evaluar el data drift y DECIDIR si corresponde
re-entrenar. Este pipeline NO entrena: solo aplica la regla y, si detecta
drift, DISPARA el pipeline de entrenamiento.

Es disparado por el pipeline de inferencia (también puede correrse manual).
"""
from airflow import DAG
from airflow.operators.python import ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import sys

sys.path.append('/opt/airflow')

from src.monitor import check_drift_and_alerts

default_args = {
    'owner': 'mlops_group_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'pipeline_monitoreo',
    default_args=default_args,
    description='Pipeline de monitoreo: evalúa drift y dispara el re-entrenamiento si es necesario',
    schedule_interval=None,  # Disparado por el pipeline de inferencia
    catchup=False,
) as dag:

    # Regla de decisión: check_drift_and_alerts devuelve True (hay drift) o False.
    # Si es True, el ShortCircuit deja continuar y se dispara el entrenamiento.
    task_check_drift = ShortCircuitOperator(
        task_id='monitorear_y_evaluar_drift',
        python_callable=check_drift_and_alerts,
    )

    # Solo se ejecuta si hubo drift: dispara el pipeline de entrenamiento
    task_trigger_entrenamiento = TriggerDagRunOperator(
        task_id='disparar_reentrenamiento',
        trigger_dag_id='pipeline_entrenamiento',
        reset_dag_run=True,
    )

    task_check_drift >> task_trigger_entrenamiento
