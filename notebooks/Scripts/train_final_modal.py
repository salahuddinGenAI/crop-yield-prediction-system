"""
train_final_modal.py
Retrain the final model pipeline and saves it.
Run this pipeline in the activated venv only.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import sklearn

print("Using scikit-learn version:", sklearn.__version__)

# Loading cleaned dataset
df = pd.read_csv("Dataset/processed/crop_yield_cleaned.csv")
df["Fertilizer_Used"] = df["Fertilizer_Used"].fillna("None")

# Defining feature groups 
numeric_features = ["Rainfall_mm", "Temperature_C", "Humidity_%"]
categorical_features = ["Soil_Type", "Crop_Type", "Fertilizer_Used", "Region"]

# Building preprocessing + model pipelining

preprocessor = ColumnTransformer(
    transformers= [
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
    ],
    remainder="passthrough"

)

best_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=5,
    min_samples_split=5,
    max_features= 0.8,
    random_state=42
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])

# ======================================
#           Data Preparation
# ======================================
# # Irrigation_Status must be 0/1 before this step
df["Irrigation_Status"] = df["Irrigation_Status"].map({"Yes": 1, "No": 0})

x_full =  df.drop(columns=["Crop_Yield_tons_per_hectare"])
y_full = df["Crop_Yield_tons_per_hectare"]

# Training on full Dataset
final_pipeline.fit(x_full, y_full),
print(f"Model trained on {x_full.shape} rows")

# Sanity Check
sample_pred = final_pipeline.predict(x_full.iloc[:5])
print(f"Sample predictions: {sample_pred}")
print(f"Actual values: {y_full.iloc[:5].values}")

# Saving or serializing of model
joblib.dump(final_pipeline, "crop_yield_model.pkl")
print("Saved crop_yield_model.pkl Successfully")

