import pandas as pd
import numpy as np
import os
from datetime import datetime
from utils import pipe
from data_preparation import add_datetime_features, encode_categoricals


# Constants
PREDICTION_INPUT_PATH = "focos_diario_br_20250520.csv"
DAY_BEFORE_INPUT_PATH = "focos_diario_br_20250519.csv"
OUTPUT_PATH = "prediction_data_prepared.parquet"
REFERENCE_HISTORY_PATH = "prepared_data/data_prepared.parquet"  
DATE_COLUMN = "data_hora_gmt"



def load_single_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, parse_dates=[DATE_COLUMN])

def prepare_daily_prediction_data():
    df_today = load_single_file(PREDICTION_INPUT_PATH)
    df_today = pipe(
        df_today,
        add_datetime_features,
        get_day_before_features,
        encode_categoricals,
        get_model_columns,
        fill_null_values)
    
    return df_today


def load_single_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, parse_dates=[DATE_COLUMN])


def load_single_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, parse_dates=[DATE_COLUMN])


def get_day_before_data():
    df_day_before = load_single_file(DAY_BEFORE_INPUT_PATH)

    features_to_merge = [ "estado", "municipio"]

    df_latest = df_day_before.groupby(features_to_merge, as_index=False).agg(
    numero_dias_sem_chuva_day_before = ("numero_dias_sem_chuva", "mean"),
    precipitacao_day_before = ("precipitacao", "mean")
    )
    return df_latest
    

def get_day_before_features(df_today):
    df_features_day_before = get_day_before_data()
    return df_today.merge(df_features_day_before, on = ['estado', 'municipio'], how='left').fillna(0)
    
    

def get_model_columns(df_today):
    df_model = pd.read_parquet(REFERENCE_HISTORY_PATH)

    model_columns = df_model.columns
        
    # Add missing columns with zero
    for col in model_columns:
        if col not in df_today and df_model[col].dtype == 'bool':
            df_today[col] = False
        elif col not in df_today :
            df_today[col] = 0

    return df_today[model_columns].drop(columns=['risco_fogo'])
    
def fill_null_values(df_today):
    return df_today.fillna(0)
    