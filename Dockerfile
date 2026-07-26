# Versión fija de la imagen base de Airflow con Python específico
FROM apache/airflow:2.7.2-python3.10

USER airflow

# Instalación de paquetes con versiones estrictamente fijadas
RUN pip install --no-cache-dir \
    mlflow==2.10.2 \
    scikit-learn==1.4.0 \
    xgboost==2.0.3 \
    scipy==1.12.0 \
    requests==2.31.0