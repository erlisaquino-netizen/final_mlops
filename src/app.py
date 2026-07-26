import streamlit as st
import pandas as pd
import numpy as np
import mlflow.sklearn
from src.config import MLFLOW_TRACKING_URI

# 🛠️ Configuración inicial para limpiar la interfaz
st.set_page_config(
    page_title="Tablero de Inferencia - MLOps",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)
  
# Configurar MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

st.title("🚀 Tablero de Inferencia: Modelos de Propensión y Copamiento")
st.markdown("Este panel carga los modelos campeones registrados en **MLflow** y ejecuta la inferencia encadenada.")

@st.cache_resource
def cargar_modelos():
    try:
        model_m1 = mlflow.sklearn.load_model("models:/Propension_Modelo_Campeon/latest")
        model_m2 = mlflow.sklearn.load_model("models:/Copamiento_Modelo_Campeon/latest")
        return model_m1, model_m2
    except Exception as e:
        st.error(f"Error al cargar modelos desde MLflow: {e}")
        return None, None

model_m1, model_m2 = cargar_modelos()

if model_m1 and model_m2:
    st.success("✅ ¡Modelos campeones cargados correctamente desde MLflow!")

    st.sidebar.header("Opciones de Entrada")
    opcion = st.sidebar.radio("Selecciona el método de prueba:", ["Subir archivo CSV de clientes", "Simular clientes con datos aleatorios"])

    df_entrada = None

    if opcion == "Subir archivo CSV de clientes":
        archivo_subido = st.sidebar.file_uploader("Sube tu archivo CSV con características de clientes", type=["csv"])
        if archivo_subido is not None:
            df_entrada = pd.read_csv(archivo_subido)
            st.write(f"📁 **Datos del archivo cargado exitosamente ({len(df_entrada)} registros):**")
            st.dataframe(df_entrada.head())
    else:
        st.sidebar.info("Generando datos de prueba aleatorios basados en las columnas del modelo M1.")
        if st.sidebar.button("Generar nuevos datos aleatorios"):
            try:
                features_m1 = model_m1.feature_names_in_
                df_entrada = pd.DataFrame(np.random.randn(5, len(features_m1)), columns=features_m1)
            except AttributeError:
                df_entrada = pd.DataFrame({
                    'edad': [35, 45, 28, 50, 40],
                    'ingresos': [50000, 120000, 25000, 90000, 60000],
                    'monto_ofertado': [10000, 20000, 5000, 15000, 12000]
                })

    if df_entrada is not None:
        st.markdown("---") 
        st.subheader("⚙️ Ejecutando Inferen...")

        # 🎛️ Control deslizante para ajustar el umbral de decisión del Modelo 1
        umbral_propension = st.sidebar.slider(
            "Umbral de Probabilidad para Modelo 1", 
            min_value=0.0, max_value=1.0, value=0.05, step=0.01,
            help="Baja este umbral para capturar más clientes aptos si el modelo es muy restrictivo."
        )

        # Preprocesamiento M1: Alinear automáticamente las columnas del CSV con las que exige el modelo
        if hasattr(model_m1, 'feature_names_in_'):
            expected_features_m1 = model_m1.feature_names_in_
            X_m1 = df_entrada.reindex(columns=expected_features_m1, fill_value=0)
        else:
            X_m1 = df_entrada

        X_m1 = X_m1.fillna(0)

        # Predicción M1
        df_entrada['prob_venta'] = model_m1.predict_proba(X_m1)[:, 1]
        
        # Aplicar el umbral personalizado definido en el slider
        df_entrada['pred_venta'] = (df_entrada['prob_venta'] >= umbral_propension).astype(int)

        # Predicción M2 (Condicional)
        df_entrada['pred_copamiento'] = 0.0
        clientes_aptos_idx = df_entrada[df_entrada['pred_venta'] == 1].index

        if len(clientes_aptos_idx) > 0:
            X_m2 = df_entrada.loc[clientes_aptos_idx]
            if hasattr(model_m2, 'feature_names_in_'):
                X_m2 = X_m2.reindex(columns=model_m2.feature_names_in_, fill_value=0)
            
            preds_log = model_m2.predict(X_m2)
            df_entrada.loc[clientes_aptos_idx, 'pred_copamiento'] = np.expm1(preds_log)

        # Mostrar resultados visuales
        st.write("📊 **Resultados Finales:**")
        
        # Destacar métricas rápidas
        total_clientes = len(df_entrada)
        aptos_venta = int(df_entrada['pred_venta'].sum())
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Clientes Evaluados", total_clientes)
        col2.metric("Clientes Aptos (Modelo 1 = 1)", aptos_venta)
        col3.metric("Monto Total Estimado (Modelo 2)", f"${df_entrada['pred_copamiento'].sum():,.2f}")

        # Mostrar tabla interactiva completa
        st.dataframe(df_entrada)

        # Botón para descargar resultados
        csv = df_entrada.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar resultados en CSV",
            data=csv,
            file_name="resultados_inferencia.csv",
            mime="text/csv",
        ) 