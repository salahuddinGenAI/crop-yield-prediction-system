# 🌾 Crop Yield Prediction System

A machine learning system that predicts expected crop yield (tons/hectare) from environmental and farming conditions — built end-to-end from synthetic data generation through EDA, model comparison, and a deployed Streamlit interface.

![Python](https://img.shields.io/badge/Python-3.12.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

Agricultural productivity depends on a complex interplay of environmental conditions and farming practices. This project builds a **regression-based ML pipeline** that predicts crop yield using rainfall, temperature, humidity, soil type, crop type, fertilizer choice, irrigation status, and region — then wraps the trained model in an interactive web app for real-time predictions.

[**Live demo:**] _(https://crop-yield-prediction-system-nscpcuveeq7e4y5olvjo2r.streamlit.app/)_

---

## ✨ Features

- 📊 Full exploratory data analysis with univariate, bivariate, and correlation insights
- 🧹 Robust data cleaning pipeline (handles missing values, encoding edge cases)
- 🤖 Three regression models compared: Linear Regression, Decision Tree, Random Forest
- 🎯 Cross-validated model evaluation (RMSE, MAE, R²)
- 💾 Serialized end-to-end pipeline (preprocessing + model bundled together)
- 🖥️ Interactive Streamlit UI with:
  - Real-time yield prediction
  - Feature importance visualization
  - Historical comparison against typical crop yield ranges
  - Input validation warnings for out-of-range values

---

## 🖼️ Screenshot

![App Screenshot]("F:\Machine Learning projects\Crop Yield Prediction System\images\Crop-Yield-Predictor-07-30-2026_11_57_PM.png")

---

## 🧠 Problem Statement

Predict the expected crop yield based on environmental and farming conditions, enabling data-driven decisions for farmers, agribusinesses, and agricultural planners.

| | |
|---|---|
| **ML Type** | Regression |
| **Target** | Crop Yield (tons/hectare) |
| **Models** | Linear Regression, Decision Tree Regressor, Random Forest Regressor |

---

## 📂 Dataset

A synthetically generated dataset (5,000 records) with domain-informed, non-linear relationships between features and yield — including realistic "sweet spot" effects (e.g., yield peaks at moderate rainfall and declines at both extremes).

| Feature | Type | Description |
|---|---|---|
| `Rainfall_mm` | Numeric | Seasonal rainfall (mm) |
| `Temperature_C` | Numeric | Average temperature (°C) |
| `Humidity_%` | Numeric | Average relative humidity (%) |
| `Soil_Type` | Categorical | Loamy, Sandy, Clay, Silty, Peaty, Unknown |
| `Crop_Type` | Categorical | Wheat, Rice, Maize, Sugarcane, Cotton, Soybean |
| `Fertilizer_Used` | Categorical | Urea, DAP, Compost, NPK, None |
| `Irrigation_Status` | Categorical | Yes / No |
| `Region` | Categorical | North, South, East, West, Central |
| `Crop_Yield_tons_per_hectare` | Numeric (Target) | Yield in tons per hectare |

---

## 🔍 Key EDA Insight

Environmental features show **near-zero linear correlation** with yield (Rainfall: 0.01, Temperature: -0.02) — but this is misleading. Per-crop scatter plots reveal a strong **non-linear "sweet spot" relationship**: yield rises with rainfall up to an optimum (~750mm), then declines. This is why tree-based models (Random Forest) substantially outperform Linear Regression on this dataset.

---

## 🏗️ Project Structure

```
crop-yield-prediction-system/
├── app/
│   ├── app.py                  # Streamlit UI
│   └── predict.py              # Inference script
├── data/
│   ├── raw/
│   │   └── crop_yield_data.csv
│   └── processed/
│       └── crop_yield_cleaned.csv
├── notebooks/
│   ├── 00_project_introduction.md
│   ├── 01_eda.py
│   ├── 02_eda_bivariate.py
│   ├── 03_eda_correlation_outliers.py
│   ├── 04_eda_summary.md
│   └── eda_plots/
├── src/
│   ├── generate_dataset.py     # Synthetic dataset generator
│   └── train_final_model.py    # Final model training script
├── crop_yield_model.pkl        # Trained pipeline (excluded from repo — see below)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📈 Model Performance

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression (baseline) | 2.483607  | 4.645891   | 0.963901 |
| Decision Tree Regressor | 1.512094   | 3.727389   | 0.976764 |
| Random Forest Regressor | 1.425270   | 3.564084   | 0.978755 |


**Selected model:** Random Forest Regressor — chosen for its ability to capture the non-linear rainfall/temperature relationships identified during EDA.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11 or 3.12 recommended

### Installation

```bash
git clone https://github.com/<your-username>/crop-yield-prediction-system.git
cd crop-yield-prediction-system

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### Regenerate the dataset (optional — a version is already included)
```bash
cd src
python generate_dataset.py
```

### Train the final model
```bash
python src/train_final_model.py
```

### Run the app
```bash
cd app
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Data & Analysis:** pandas, numpy, matplotlib, seaborn
- **Modeling:** scikit-learn, joblib
- **UI:** Streamlit

---

## 🔮 Future Improvements

- Deploy to Streamlit Community Cloud / HuggingFace Spaces
- Add hyperparameter tuning results/logs (GridSearchCV) to the repo
- Expand to real-world agricultural datasets
- Add confidence intervals to predictions
- Per-crop specialized models

---

## 📝 Adding a Screenshot

1. Run the app locally and take a screenshot
2. Create a `docs/` folder in your repo and save it as `docs/screenshot.png`
3. The image reference in this README will pick it up automatically

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙋 Author

Built by Salahuddin Ayubi as part of an AI/ML learning project series.

_Connect: [LinkedIn](https://www.linkedin.com/in/salahud-din-ayubi/) 
          [GitHub](https://github.com/salahuddinGenAI)_
