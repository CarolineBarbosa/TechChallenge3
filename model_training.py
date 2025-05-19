import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models import get_models
import joblib

INPUT_FILE = "prepared_data/data_prepared.parquet"


def select_model_pipeline():
    df = load_prepared_data(INPUT_FILE)
    df = prepare_data(df)
    X_train, X_val, y_train, y_val = split_train_val(df)
    models = get_models()
    best_model, best_name = find_best_model(models, X_train, X_val, y_train, y_val)
    save_best_model(best_model, best_name)
    # train_model( X_train, X_val, y_train, y_val)

def load_prepared_data(folder_path):
    df = pd.read_parquet(folder_path)
    return df

def prepare_data(df):
    df = df.fillna(0)
    return df

def split_train_val(df):
    df = df.sort_values(by="day_of_year")
    unique_days = df["day_of_year"].unique()
    split_day = sorted(unique_days)[-30]  # Last month = last 30 days
    train_df = df[df["day_of_year"] < split_day]
    val_df = df[df["day_of_year"] >= split_day]

    X_train = train_df.drop(columns=["risco_fogo"])
    y_train = train_df["risco_fogo"]
    X_val = val_df.drop(columns=["risco_fogo"])
    y_val = val_df["risco_fogo"]

    return X_train, X_val, y_train, y_val


def evaluate_model(name, model, X_val, y_val, X_train, y_train):
    y_pred = model.predict(X_val)
    ypred_1 = model.predict(X_train)
    print(f"🔍 {name}")
    print(f"MAE_val: {mean_absolute_error(y_val, y_pred):.4f}")
    print(f"RMSE_val: {mean_squared_error(y_val, y_pred):.4f}")
    print(f"MAE_train: {mean_absolute_error(y_train, ypred_1):.4f}")
    print(f"RMSE_train: {mean_squared_error(y_train, ypred_1):.4f}")
    print(f"R² Score_val: {r2_score(y_val, y_pred):.4f}")
    print("-" * 40)
    return mean_squared_error(y_val, y_pred), model

def train_model(X_train, X_val, y_train, y_val):
    models = get_models()
    best_model, best_name = find_best_model(models, X_train, X_val, y_train, y_val)
    save_best_model(best_model, best_name)


def find_best_model(models, X_train, X_val, y_train, y_val):
    best_metric = float("inf")
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        metric, trained_model = evaluate_model(name, model, X_val, y_val, X_train, y_train)
        if metric < best_metric:
            best_metric = metric
            best_model = trained_model
            best_name = name

    return best_model, best_name


def save_best_model(model, model_name):
    file_name = "fire_risk_model_v1.pkl"
    joblib.dump(model, file_name)
    print(f"✅ Best model ({model_name}) saved to {file_name}")

if __name__ == "__main__":
    select_model_pipeline()