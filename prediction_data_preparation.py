import pandas as pd
import numpy as np
import os
from datetime import datetime
from utils import pipe
from data_preparation import add_datetime_features, encode_categoricals


# Constants

OUTPUT_PATH = "prediction_data_prepared.parquet"
REFERENCE_HISTORY_PATH = "prepared_data/data_prepared.parquet"  
DATE_COLUMN = "data_hora_gmt"



def load_single_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, parse_dates=[DATE_COLUMN])

def prepare_daily_prediction_data(reference_date: str = None, yesterday_date: str = None) -> pd.DataFrame:
    PREDICTION_INPUT_PATH = os.path.join("daily_data", f"focos_diario_br_{reference_date}.csv")
    DAY_BEFORE_INPUT_PATH = os.path.join("daily_data", f"focos_diario_br_{yesterday_date}.csv")
    
    
    df_today = load_single_file(PREDICTION_INPUT_PATH)
    df_today = pipe(
        df_today,
        add_datetime_features,
        lambda df: get_day_before_features(df, DAY_BEFORE_INPUT_PATH),
        encode_categoricals,
        get_model_columns,
        fill_null_values)
    
    return df_today


def load_single_file(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path, parse_dates=[DATE_COLUMN])


def get_day_before_data(filepath: str = None) -> pd.DataFrame:
    df_day_before = load_single_file(filepath)

    features_to_merge = [ "estado", "municipio"]

    df_latest = df_day_before.groupby(features_to_merge, as_index=False).agg(
    numero_dias_sem_chuva_day_before = ("numero_dias_sem_chuva", "mean"),
    precipitacao_day_before = ("precipitacao", "mean")
    )
    return df_latest
    

def get_day_before_features(df_today, filepath):
    df_features_day_before = get_day_before_data(filepath)
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
    