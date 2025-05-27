import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models import get_models
import joblib

INPUT_FILE = "prepared_data/data_prepared.parquet"
MODEL_OUTPUT_PATH = "model/fire_risk_model_v2.joblib"


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
    unique_days = df[df.year==2025]["day_of_year"].unique()
    split_day = sorted(unique_days)[-30]  # Last month = last 30 days
    train_df = df[~((df["day_of_year"] >= split_day) & (df["year"] == 2025))]
    val_df = df[(df["day_of_year"] >= split_day) & (df["year"] == 2025)]
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Train unique days: {train_df['day_of_year'].nunique()}")
    print(f"Validation unique days: {val_df['day_of_year'].nunique()}")
    X_train = train_df.drop(columns=["risco_fogo"])
    y_train = train_df["risco_fogo"]
    X_val = val_df.drop(columns=["risco_fogo"])
    y_val = val_df["risco_fogo"]

    return X_train, X_val, y_train, y_val

def evaluate_model(name, model, X_val, y_val, X_train, y_train):
    y_pred = model.predict(X_val)
    ypred_1 = model.predict(X_train)
    results = pd.DataFrame([{
        "model_name": name,
        "mae_val": mean_absolute_error(y_val, y_pred),
        "rmse_val": mean_squared_error(y_val, y_pred),
        "mae_train": mean_absolute_error(y_train, ypred_1),
        "rmse_train": mean_squared_error(y_train, ypred_1),
        "r2_score_val": r2_score(y_val, y_pred),
    }])
    return results, model

def train_model(X_train, X_val, y_train, y_val):
    models = get_models()
    best_model, best_name = find_best_model(models, X_train, X_val, y_train, y_val)
    save_best_model(best_model, best_name)

def find_best_model(models, X_train, X_val, y_train, y_val):
    best_metric = float("inf")
    best_model = None
    best_name = ""

    models_results = pd.DataFrame(columns=["model_name", "mae_val", "rmse_val", "mae_train", "rmse_train", "r2_score_val"])
    for name, model in models.items():
        print(name)
        model.fit(X_train, y_train)
        results, trained_model = evaluate_model(name, model, X_val, y_val, X_train, y_train)
       
        metric = results["rmse_val"][0]
        if metric < best_metric:
            best_metric = metric
            best_model = trained_model
            best_name = name
        models_results = pd.concat([models_results, results])
    models_results.to_csv("model_results.csv", index=False)
    return best_model, best_name


def save_best_model(model, model_name):
    joblib.dump(model, MODEL_OUTPUT_PATH)
    print(f"✅ Best model ({model_name}) saved to {MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    select_model_pipeline()