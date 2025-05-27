import joblib
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, File
import numpy as np
import pandas as pd
from scrapper import scrape_and_collect_data, ingest_data
from prediction_data_preparation import prepare_daily_prediction_data
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
import os


router = APIRouter()

# Carregar modelo diretamente no controller.py
with open("model/fire_risk_model_v1.joblib", "rb") as f:
    modelo = joblib.load(f)

class InputData(BaseModel):
    features: list


@router.get("/get_data/")
async def get_data(date: str = None):
    """
    Endpoint to fetch the latest fire risk data.
    """
    try:
        formatted_date = pd.to_datetime(date, format="%d-%m-%Y")
    except ValueError:
        return {"error": "Invalid date format. Use dd-mm-yyyy."}
    
    formatted_date_str = formatted_date.strftime("%d-%m-%Y")
    reference_date_formatted = "".join(reversed(formatted_date_str.split("-")))

    try:
        # Call the scrape_and_collect_data() function to fetch data
        scrape_and_collect_data(reference_date_formatted)
        ingest_data()
        return {"message": "Data fetched successfully."}
    
    except Exception as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

@router.post("/predict/")
async def predict(date: str):
    """
    Endpoint to predict fire risk based on a given date.
    """
    # Validate the date format
    try:
        formatted_date = pd.to_datetime(date, format="%d-%m-%Y")
    except ValueError:
        return {"error": "Invalid date format. Use dd-mm-yyyy."}

    # Call the scrape_and_collect_data() function to fetch data
    formatted_date_str = formatted_date.strftime("%d-%m-%Y")
    yesterday_date = (formatted_date - pd.Timedelta(days=1)).strftime("%d-%m-%Y")

    reference_date_formatted = "".join(reversed(formatted_date_str.split("-")))
    yesterday_date_formatted = "".join(reversed(yesterday_date.split("-")))

    try:
        scrape_and_collect_data(reference_date_formatted)
        scrape_and_collect_data(yesterday_date_formatted)
    except Exception as e:
        return {"error": f"Failed to fetch data: {str(e)}"}

    # Prepare the data for prediction
    try:
        df_today = prepare_daily_prediction_data(reference_date_formatted, yesterday_date_formatted)
    except Exception as e:
        return {"error": f"Failed to prepare data: {str(e)}"}
    # Ensure the model is loaded and ready for prediction
    if modelo is None:
        return {"error": "Model not loaded."}
    # Make predictions
    try:
        prediction = modelo.predict(df_today)
        print(f"Prediction shape: {prediction.shape}")
        df_output = pd.read_csv(f'daily_data/focos_diario_br_{reference_date_formatted}.csv')
        df_output["risco_fogo"] = prediction
        print(f"Output shape: {df_output.shape}")
        df_output = df_output[[
            "id", "lat", "lon", "data_hora_gmt", "satelite", "municipio", "estado", 
            "pais", "municipio_id", "estado_id", "pais_id", "numero_dias_sem_chuva", 
            "precipitacao", "risco_fogo", "bioma", "frp"
        ]]
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
        # Return the head(10) of the dataset and a download option
    try:
        
        # Save the full dataset to a CSV file for download
        output_file_path = f"focos_diario_br_{reference_date_formatted}_output.csv"
        # df_output.to_csv(output_file_path, index=False)
        return StreamingResponse( iter([df_output.to_csv(index=False)]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={output_file_path}.csv"}
            )

        
    except Exception as e:
        return {"error": f"Failed to process output: {str(e)}"}

@router.get("/metricas")
def download_csv():
    caminho_arquivo = "model_results.csv"

    # Verificação de arquivo
    if not os.path.exists(caminho_arquivo):
        return {"erro": "Arquivo não encontrado"}

    # Retorna o arquivo para download
    return FileResponse(
        path=caminho_arquivo,
        media_type='text/csv',
        filename='metricas.csv'
    )
