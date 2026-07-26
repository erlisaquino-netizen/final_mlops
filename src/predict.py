import pandas as pd
import mlflow
import os

# Ajusta estas rutas a las carpetas locales de tu proyecto si no estás ejecutando dentro del contenedor
#NUEVOS_DATOS_PATH = "data/raw_data/nuevos_clientes.csv" 
#OUTPUT_PRED_PATH = "data/processed/predicciones_propension.csv"
   
NUEVOS_DATOS_PATH = "/opt/airflow/raw_data/nuevos_clientes.csv"
OUTPUT_PRED_PATH = "/opt/airflow/raw_data/processed/predicciones_propension.csv"

# Tu URI de MLflow y el Run ID del modelo real (0.8345)
#mlflow.set_tracking_uri("http://localhost:5000") # Asegura que apunte a tu MLflow local
 
mlflow.set_tracking_uri("http://mlflow:5000")
logged_model = 'runs:/1d49dd2fc54e4701b62bacda7c14eb1d/model'

def run_inference():
    print("Cargando nuevos datos de clientes...")
    if not os.path.exists(NUEVOS_DATOS_PATH):
        print(f"❌ Error: No se encontró el archivo en {NUEVOS_DATOS_PATH}. Crea el archivo primero.")
        return
        
    df_nuevos = pd.read_csv(NUEVOS_DATOS_PATH)
    
    # Columnas que debemos quitar para que el modelo no se rompa (mismo criterio del entrenamiento)
    columnas_a_excluir = ['target_venta', 'target_copamiento', 'id_cliente', 'id', 'codunicocli', 'copamiento', 'monto', 'monto_aceptado']
    X_nuevos = df_nuevos.drop(columns=columnas_a_excluir, errors='ignore')
    
    print("Cargando el modelo campeón desde MLflow...")
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    
    print("Calculando probabilidades de propensión...")
    # Usamos predict_proba para obtener la probabilidad decimal (ej. 0.85 de probabilidad de compra)
    # Si tu modelo cargado con pyfunc no soporta predict_proba directamente, predict te dará la clase (0 o 1)
    try:
        #predicciones = loaded_model.unwrap_python_model().predict_proba(X_nuevos)[:, 1]

        print("Calculando probabilidades de propensión...")
         
        # MLflow pyfunc por defecto usa .predict(), que para clasificadores de sklearn suele retornar las clases (0 o 1).
        # Como la gran mayoría de modelos en MLflow pyfunc devuelven la clase o la probabilidad directamente según cómo fue logueado,
        # ejecutamos la predicción estándar:
        predicciones = loaded_model.predict(X_nuevos)

 
    except AttributeError:
        predicciones = loaded_model.predict(X_nuevos)
    
    # Agregamos la columna al DataFrame original para saber a quién corresponde cada predicción
    df_nuevos['propension_compra'] = predicciones
    
    # Guardamos el resultado final
    os.makedirs(os.path.dirname(OUTPUT_PRED_PATH), exist_ok=True)
    df_nuevos.to_csv(OUTPUT_PRED_PATH, index=False)
    print(f"✅ ¡Inferencia completada con éxito! Resultados guardados en: {OUTPUT_PRED_PATH}")

if __name__ == "__main__":
    run_inference()