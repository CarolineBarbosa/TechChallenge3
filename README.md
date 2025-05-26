# TechChallenge 3
This project was developed as part of the FIAP Machine Learning Engineering course. The objective of this challenge was to explore the applicability of a machine learning model using real-world data. It involved analyzing data, building predictive models, and evaluating their performance to solve practical problems.

## Project Structure

The project is organized into the following structure:

```
TechChallenge3/
├── data/               # Contains raw and processed datasets
├── notebooks/          # Jupyter notebooks for EDA and model development
├── src/                # Source code for data processing, model training, and API
├── models/             # Saved machine learning models
├── prepared_data/      # Saved prepared data
├── daily_data/         # Saved daily data
├── api/                # API implementation for predictions
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Workflow

1. **Data Extraction**:  
    The initial dataset was sourced from a public repository providing open data on fire risk in Brazil:  
    [TerraBrasilis - INPE](https://terrabrasilis.dpi.inpe.br/queimadas/portal/dados-abertos/#da-focos).  

    This dataset includes +2 years of information related to fire risk across various regions in Brazil. The features in the dataset are as follows:  

    - `id`: Unique identifier for each record  
    - `lat`: Latitude of the fire occurrence  
    - `lon`: Longitude of the fire occurrence  
    - `data_hora_gmt`: Date and time in GMT  
    - `satelite`: Satellite used for data collection  
    - `municipio`: Municipality where the fire occurred  
    - `estado`: State where the fire occurred  
    - `pais`: Country where the fire occurred  
    - `municipio_id`: Municipality identifier  
    - `estado_id`: State identifier  
    - `pais_id`: Country identifier  
    - `numero_dias_sem_chuva`: Number of days without rain  
    - `precipitacao`: Precipitation levels  
    - `risco_fogo`: Fire risk level  (Target)
    - `bioma`: Biome classification  
    - `frp`: Fire Radiative Power  

    This rich dataset served as the foundation for building and evaluating the machine learning models in this project.
    
2. **Exploratory Data Analysis (EDA)**:  
    We performed an in-depth analysis of the data to understand its structure, identify patterns, and handle missing or inconsistent values. Details regarding the Data analysis can found in the `notebooks/EDA.ipynb` file. 


3. **Data Preparation**:  
    The dataset had a comprehensive cleaning and transformation process to ensure its suitability for model training. This included handling missing values, feature engineering, and scaling. Details of the transformations applied can be found in the `data_preparation.py` file.

4. **Model Training**:  
    Various machine learning models were trained and evaluated to predict the risk of fire. The best-performing model was selected based on rmse and other performance metrics. This model was the Gradient Boosting model, which, despite its limitations, demonstrated the highest performance among the tested models. 

    The model achieved an RMSE of 0.08  for fire risk prediction. While this performance is not ideal, it was deemed the most suitable for the task given the available data and constraints. The training process utilized over two years of historical data, while the last month of data was reserved for validation. 

    We chose to use only one month for validation to ensure that the model's performance was assessed on the most recent and relevant data, reflecting potential seasonal variations and trends. This approach allowed us to evaluate the model's ability to generalize to unseen data while maintaining a realistic and practical validation framework.

    To further enhance the model's accuracy, incorporating additional data sources related to climate or vegetation could be explored in the future. For now, the selected model provides a solid foundation for fire risk prediction. The plan could be to retrain the model on a monthly basis to ensure it remains updated and capable of delivering the most accurate predictions based on the latest data.

5. **API Development**:  
    The final model was integrated into a RESTful API, allowing users to make predictions on the daily fire risk by providing input data.

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/your-repo/TechChallenge3.git
    cd TechChallenge3
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the API:
    ```bash
    python -m uvicorn main:app
    ```
4. Access the API documentation:  
    Open your browser and navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to explore the interactive API interface.

5. Use the API to make predictions by sending a GET or POST request with the required input data.

### API Endpoints

The API provides two main endpoints:

1. **GET Endpoint**:  
    This endpoint allows users to extract data for a specified date. The extracted data is saved as a file in the `daily_data/` folder for further use.

2. **POST Endpoint**:  
    This endpoint offers a more comprehensive workflow. It requires a reference date as input and performs the following steps:  
    - Checks if the dataset for the specified date exists in the `daily_data/` folder.  
    - If the dataset is not found, it fetches the data from the source website.  
    - Applies data preparation steps to ensure the data is ready for predictions.  
    - Makes fire risk predictions for each requested region.  
    - Generates a downloadable file containing the predictions, which is made available to the user.

These endpoints enable seamless data extraction, preparation, and prediction workflows, making the API a powerful tool for fire risk analysis.


## Conclusion

This project demonstrates the end-to-end process of developing a machine learning solution, from data extraction and analysis to model deployment via an API. It highlights the practical application of machine learning in solving real-world problems.