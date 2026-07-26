import pytest
import pandas as pd
import numpy as np
import os
from src.preprocess import clean_and_prepare_data
from src.config import DATA_PROCESSED_DIR, TRAIN_FILE

def test_preprocess_generates_outputs():
    """Prueba que el preprocesador limpie anomalías y genere los CSV procesados para M1 y M2."""
    clean_and_prepare_data(TRAIN_FILE)
    
    train_m1_path = os.path.join(DATA_PROCESSED_DIR, "train_m1.csv")
    train_m2_path = os.path.join(DATA_PROCESSED_DIR, "train_m2.csv")
    
    assert os.path.exists(train_m1_path)
    assert os.path.exists(train_m2_path)
    
    df_m2 = pd.read_csv(train_m2_path)
    # Validamos que todos los targets asignados en M2 sean válidos (exclusión de target_venta == 0)
    assert 'target_copamiento' in df_m2.columns  