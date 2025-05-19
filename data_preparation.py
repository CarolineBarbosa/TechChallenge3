import pandas as pd
import numpy as np
import os
from typing import List
from utils import pipe

# Constants
INPUT_FOLDER = "data/"
OUTPUT_PATH = "prepared_data/data_prepared.parquet"
DATE_COLUMN = "data_hora_gmt"
DATE_CUTOFF = "2023-01-01"


def prepare_training_data():

    df = pipe(
        load_from_folder(INPUT_FOLDER),
        add_datetime_features,
        filter_valid_values,
        fill_and_engineer_features,
        encode_categoricals,
        filter_columns,
    )
    export_filtered_data(df, OUTPUT_PATH, DATE_CUTOFF)


# def prepare_training_data():
#     df = load_data(INPUT_FOLDER)
#     df = add_datetime_features(df)
#     df = df.drop(['data_hora_gmt', 'pais', 'pais_id'], axis=1)
#     df = df[df['risco_fogo'] > 0]
#     df = fill_and_engineer_features(df)
#     df = encode_categoricals(df)
#     df = filter_columns(df)
#     export_filtered_data(df, OUTPUT_PATH, DATE_CUTOFF)


def load_from_folder(folder_path: str) -> pd.DataFrame:
    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".csv")])
    df_list = [
        pd.read_csv(os.path.join(folder_path, file), parse_dates=["data_hora_gmt"])
        for file in all_files
    ]
    df = pd.concat(df_list, ignore_index=True)
    return df


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = df[DATE_COLUMN].dt.floor("D")
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.weekday
    df["year"] = df["date"].dt.year
    df["cos_day_of_year"] = encode_day_of_year_cyclic(df["day_of_year"])
    return df


def encode_day_of_year_cyclic(day_of_year):
    """
    Encodes the day of the year (1–366) using a cosine transformation to capture its cyclical nature.

    This approach is useful because days of the year are cyclical — day 1 and day 365 are
    temporally adjacent, but numerically far apart. Traditional models don't account for this
    cyclical relationship. By projecting the day onto a unit circle using a cosine function,
    we preserve the seasonal structure and continuity of the time variable.

    The transformation maps each day to a point on a circle, where similar times of the year
    produce similar values, making it easier for machine learning models to learn seasonal patterns.

    Parameters:
        day_of_year (int): The day of the year, ranging from 1 to 366.

    Returns:
        float: Cosine-encoded value representing the cyclical position of the day.
    """
    return np.cos(2 * np.pi * day_of_year / 365)


def filter_valid_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["data_hora_gmt", "pais", "pais_id"], axis=1)
    df = df[df["risco_fogo"] > 0]
    return df


def fill_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_city = create_city_features(df)
    return df.merge(df_city, on=["date", "municipio", "estado"], how="left")


def create_city_features(df: pd.DataFrame) -> pd.DataFrame:
    df1 = df.copy().drop_duplicates(subset=["estado", "municipio", "date"])
    df_filled = df1.groupby(["estado", "municipio"], as_index=False).apply(
        compute_previous_day_features
    )
    feature_cols = ["date", "municipio", "estado", "numero_dias_sem_chuva_day_before", 'precipitacao_day_before']

    df_filled = df_filled[feature_cols]
    return df_filled


def compute_previous_day_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.set_index("date").reindex(
        pd.date_range(group["date"].min(), group["date"].max())
    )
    group = group.reset_index().rename(columns={"index": "date"})

    group[["estado", "municipio"]] = group[["estado", "municipio"]].ffill()
    group["numero_dias_sem_chuva_day_before"] = group["numero_dias_sem_chuva"].shift(1)
    group["precipitacao_day_before"] = group["precipitacao"].shift(1)

    return group


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=["bioma", "estado"])


def filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_export = [
        "lat",
        "lon",
        "risco_fogo",
        'date',
        "year",
        "day_of_year",
        "cos_day_of_year",
        "numero_dias_sem_chuva_day_before",
        "precipitacao_day_before",
    ]
    columns_to_export += [
        col
        for col in df.columns
        if col.startswith("bioma_") or col.startswith("estado_")
    ]
    return df[columns_to_export]


def export_filtered_data(df: pd.DataFrame, output_path: str, date_cutoff: str) -> None:
    filtered_df = df[df["date"] >= date_cutoff].drop(columns=["date"])
    filtered_df.to_parquet(output_path)


if __name__ == "__main__":
    prepare_training_data()
