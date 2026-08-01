"""
predict.py
Standalone inference script for the Crop Yield Prediction model.
Loads the saved pipeline and returns a yield prediction for new input.
"""

import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "crop_yield_model.pkl")

model_pipeline = joblib.load(MODEL_PATH)


def predict_yield(rainfall, temperature, humidity, soil_type,
                   crop_type, fertilizer_used, irrigation_status, region):
    input_df = pd.DataFrame([{
        "Rainfall_mm": rainfall,
        "Temperature_C": temperature,
        "Humidity_%": humidity,
        "Soil_Type": soil_type,
        "Crop_Type": crop_type,
        "Fertilizer_Used": fertilizer_used,
        "Irrigation_Status": 1 if irrigation_status == "Yes" else 0,
        "Region": region
    }])

    prediction = model_pipeline.predict(input_df)[0]
    return round(float(prediction), 2)


if __name__ == "__main__":
    result = predict_yield(
        rainfall=750, temperature=25, humidity=60, soil_type="Loamy",
        crop_type="Wheat", fertilizer_used="NPK", irrigation_status="Yes", region="North"
    )
    print(f"Predicted yield: {result} tons/hectare")
