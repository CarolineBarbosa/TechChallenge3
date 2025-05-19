import joblib
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File
import numpy as np
import pandas as pd

router = APIRouter()

# Carregar modelo diretamente no controller.py
with open("fire_risk_model.pkl", "rb") as f:
    modelo = joblib.load(f)

class InputData(BaseModel):
    features: list

@router.post("/predict/")
def predict(file: UploadFile = File(...)):
    # Ler o CSV enviado pelo usuário
    df = pd.read_csv(file.file)

    # Converter os dados para um array NumPy
    X = np.array(df.values)

    # Fazer a previsão
    prediction = modelo.predict(X)

    return {"prediction": prediction.tolist()}
