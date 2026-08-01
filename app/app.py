"""
app.py
Crop Yield Prediction System — Streamlit UI

Folder structure assumed:
    project_root/
    ├── crop_yield_model.pkl
    ├── data/processed/crop_yield_cleaned.csv
    └── app/
        ├── app.py        <- this file
        └── predict.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np

from predict import predict_yield, model_pipeline

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "processed", "crop_yield_cleaned.csv")

# ============================================================
# LOAD REFERENCE DATA (for comparison chart + input ranges)
# Cached so it only loads once, not on every interaction
# ============================================================
@st.cache_data
def load_reference_data():
    df = pd.read_csv(DATA_PATH)
    df["Fertilizer_Used"] = df["Fertilizer_Used"].fillna("None")  # guard against the None->NaN trap
    return df

try:
    ref_df = load_reference_data()
    DATA_AVAILABLE = True
except FileNotFoundError:
    DATA_AVAILABLE = False
    ref_df = None

# ============================================================
# EXTRACT FEATURE IMPORTANCE FROM THE LOADED PIPELINE
# Works dynamically with whatever model is inside predict.py's pipeline
# ============================================================
@st.cache_data
def get_feature_importance():
    try:
        model = model_pipeline.named_steps["model"]
        preprocessor = model_pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            return None

        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False).head(10)

        # Clean up sklearn's prefixed names (e.g. "num__Rainfall_mm" -> "Rainfall_mm")
        fi_df["Feature"] = fi_df["Feature"].str.replace(r"^(num__|cat__|remainder__)", "", regex=True)
        return fi_df
    except Exception:
        return None

feature_importance_df = get_feature_importance()

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("🌾 About This Project")
    st.markdown(
        """
        This app predicts **crop yield (tons/hectare)** from environmental
        and farming conditions using a trained machine learning model.

        Built as part of an end-to-end ML pipeline: data generation → EDA →
        preprocessing → model comparison → deployment.
        """
    )

    st.divider()

    st.header("🤖 Model Information")
    model_name = type(model_pipeline.named_steps["model"]).__name__
    st.markdown(f"**Algorithm:** {model_name}")
    st.markdown(
        """
        **Also evaluated during development:**
        - Linear Regression (baseline)
        - Decision Tree Regressor

        This model was selected as the best performer based on
        cross-validated RMSE, MAE, and R² during Milestone 5.
        """
    )

    st.divider()

    if feature_importance_df is not None:
        st.header("📊 What Drives Predictions")
        st.caption("Top features by importance in the trained model")
        st.bar_chart(feature_importance_df.set_index("Feature")["Importance"])
    else:
        st.info("Feature importance unavailable for this model type.")

    st.divider()

    st.header("💡 How to Use")
    st.markdown(
        """
        1. Fill in the environmental and farming conditions
        2. Click **Predict Yield**
        3. Compare your prediction against the typical range for that crop
        4. Values far outside the training data range will show a warning
        """
    )

    st.divider()
    st.caption("Crop Yield Prediction System · Built with Streamlit + scikit-learn")

# ============================================================
# MAIN PAGE — HEADER
# ============================================================
st.title("🌾 Crop Yield Prediction System")
st.write("Enter your farming and environmental conditions to predict expected crop yield.")

st.divider()

# ============================================================
# MAIN PAGE — INPUT FORM (two-column layout)
# ============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌱 Crop & Soil")
    crop_type = st.selectbox("Select Crop", ["Wheat", "Rice", "Maize", "Sugarcane", "Cotton", "Soybean"])
    soil_type = st.selectbox("Select Soil Type", ["Loamy", "Sandy", "Clay", "Silty", "Peaty", "Unknown"])
    region = st.selectbox("Select Region", ["North", "South", "East", "West", "Central"])

with col2:
    st.subheader("💧 Farming Practices")
    fertilizer_used = st.selectbox("Select Fertilizer", ["NPK", "DAP", "Urea", "Compost", "None"])
    irrigation_status = st.selectbox("Irrigation Status", ["Yes", "No"])

st.subheader("🌤️ Environmental Conditions")
col3, col4, col5 = st.columns(3)
with col3:
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=2000.0, value=750.0, step=10.0)
with col4:
    temperature = st.number_input("Temperature (°C)", min_value=-10.0, max_value=55.0, value=25.0, step=0.5)
with col5:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0, step=1.0)

st.divider()

# ============================================================
# PREDICTION
# ============================================================
TYPICAL_RANGES = {
    "Rainfall_mm": (200, 1500),
    "Temperature_C": (10, 45),
    "Humidity_%": (25, 90),
}

predict_clicked = st.button("🔮 Predict Yield", type="primary", use_container_width=True)

if predict_clicked:
    # --- Soft validation warnings ---
    warnings_list = []
    if not (TYPICAL_RANGES["Rainfall_mm"][0] <= rainfall <= TYPICAL_RANGES["Rainfall_mm"][1]):
        warnings_list.append(f"Rainfall ({rainfall}mm) is outside the typical training range "
                              f"({TYPICAL_RANGES['Rainfall_mm'][0]}-{TYPICAL_RANGES['Rainfall_mm'][1]}mm).")
    if not (TYPICAL_RANGES["Temperature_C"][0] <= temperature <= TYPICAL_RANGES["Temperature_C"][1]):
        warnings_list.append(f"Temperature ({temperature}°C) is outside the typical training range "
                              f"({TYPICAL_RANGES['Temperature_C'][0]}-{TYPICAL_RANGES['Temperature_C'][1]}°C).")
    if not (TYPICAL_RANGES["Humidity_%"][0] <= humidity <= TYPICAL_RANGES["Humidity_%"][1]):
        warnings_list.append(f"Humidity ({humidity}%) is outside the typical training range "
                              f"({TYPICAL_RANGES['Humidity_%'][0]}-{TYPICAL_RANGES['Humidity_%'][1]}%).")

    for w in warnings_list:
        st.warning(f"⚠️ {w} Prediction may be less reliable.")

    # --- Run prediction ---
    result = predict_yield(
        rainfall=rainfall, temperature=temperature, humidity=humidity,
        soil_type=soil_type, crop_type=crop_type, fertilizer_used=fertilizer_used,
        irrigation_status=irrigation_status, region=region
    )

    # --- Display result ---
    st.success(f"🌱 **Predicted Crop Yield: {result} tons/hectare**")

    # --- Comparison against historical data for this crop ---
    if DATA_AVAILABLE:
        crop_subset = ref_df[ref_df["Crop_Type"] == crop_type]["Crop_Yield_tons_per_hectare"]
        crop_avg = crop_subset.mean()
        crop_min = crop_subset.min()
        crop_max = crop_subset.max()

        st.subheader(f"📈 How this compares — {crop_type} typical range")

        m1, m2, m3 = st.columns(3)
        m1.metric("Your Prediction", f"{result} t/ha")
        m2.metric(f"{crop_type} Average", f"{crop_avg:.2f} t/ha",
                   delta=f"{result - crop_avg:+.2f} vs avg")
        m3.metric(f"{crop_type} Range", f"{crop_min:.1f} – {crop_max:.1f} t/ha")

        chart_df = pd.DataFrame({
            "Value": [crop_min, crop_avg, result, crop_max],
        }, index=["Min", "Average", "Your Prediction", "Max"])
        st.bar_chart(chart_df)

        if result > crop_avg:
            st.caption(f"✅ This prediction is **above average** for {crop_type} — favorable conditions.")
        else:
            st.caption(f"ℹ️ This prediction is **below average** for {crop_type} — conditions may be suboptimal "
                       f"(check rainfall/temperature against the sweet-spot range).")
    else:
        st.caption("Historical comparison unavailable — reference dataset not found at expected path.")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "⚠️ Predictions are based on a model trained on synthetic/historical data and are intended "
    "as a planning aid, not a guaranteed outcome. Actual yield depends on many additional factors."
)
