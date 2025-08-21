import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the FastAPI app
app = FastAPI(title="Sickle Cell Crisis Prediction API", version="1.0")

# 2. Load the trained model
# This assumes 'model.joblib' is in the same directory
try:
    model = joblib.load("model.joblib")
except FileNotFoundError:
    model = None # Or handle the error as you see fit

# 3. Define the input data model using Pydantic
# This ensures the input data has the correct format and features
class ModelInput(BaseModel):
    max_aod_roll6: float
    precip_range: float
    month_cos: float
    max_aod: float
    min_precipitation_roll6: float
    min_precipitation: float
    tmin_temperature: float
    month_sin: float
    tmax_temperature: float
    max_aod_lag1: float
    min_precipitation_lag1: float
    max_aod_lag3: float

# 4. Define the prediction endpoint
@app.post("/predict")
def predict(input_data: ModelInput):
    """
    Takes climate and time features as input and returns the predicted
    sickle cell crisis risk score.
    """
    if model is None:
        return {"error": "Model not loaded. Please check the model file."}

    # Convert the input data into a pandas DataFrame
    input_df = pd.DataFrame([input_data.dict()])
    
    # Ensure the column order matches the model's training data
    # (Pydantic and dict conversion should handle this, but explicit ordering is safer)
    feature_order = [
        'max_aod_roll6', 'precip_range', 'month_cos', 'max_aod', 
        'min_precipitation_roll6', 'min_precipitation', 'tmin_temperature', 
        'month_sin', 'tmax_temperature', 'max_aod_lag1', 
        'min_precipitation_lag1', 'max_aod_lag3'
    ]
    input_df = input_df[feature_order]

    # Make a prediction
    prediction = model.predict(input_df)
    
    # Clip prediction to be non-negative and round it
    final_prediction = np.round(np.clip(prediction, 0, None), 2)[0]

    return {"predicted_risk_score": float(final_prediction)}

# Optional: Add a root endpoint for basic info
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sickle Cell Crisis Prediction API. Use the /predict endpoint for predictions."}