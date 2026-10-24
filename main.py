import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os

app = FastAPI(
    title="Heart Stroke & Disease Risk Prediction API",
    description="Machine Learning Backend API for Heart Disease Prediction",
    version="1.0.0"
)

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from Next.js local server or any client
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML components
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_heart_disease.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(BASE_DIR, "columns.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    expected_cols = joblib.load(COLUMNS_PATH)
    print("Successfully loaded ML model, scaler, and expected columns.")
except Exception as e:
    print(f"Error loading ML models: {e}")
    model, scaler, expected_cols = None, None, None


class HeartPredictionInput(BaseModel):
    age: int = Field(..., ge=18, le=100, description="Age in years (18-100)")
    sex: str = Field(..., description="Sex: M or F")
    chest_pain: str = Field(..., description="Chest Pain Type: ATA, NAP, TA, ASY")
    resting_bp: int = Field(..., ge=80, le=200, description="Resting Blood Pressure (mm Hg)")
    cholesterol: int = Field(..., ge=100, le=600, description="Cholesterol (mg/dL)")
    fasting_bs: int = Field(..., ge=0, le=1, description="Fasting Blood Sugar > 120 mg/dL (1=True, 0=False)")
    resting_ecg: str = Field(..., description="Resting ECG: Normal, ST, LVH")
    max_hr: int = Field(..., ge=60, le=220, description="Max Heart Rate achieved (60-220)")
    exercise_angina: str = Field(..., description="Exercise-Induced Angina: Y or N")
    oldpeak: float = Field(..., ge=0.0, le=6.0, description="Oldpeak ST Depression (0.0 - 6.0)")
    st_slope: str = Field(..., description="ST Slope: Up, Flat, Down")


class PredictionOutput(BaseModel):
    prediction: int
    risk_label: str
    risk_level: str
    probability: float
    key_factors: List[str]
    recommendations: List[str]


@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Heart Disease Prediction API",
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_heart_disease(data: HeartPredictionInput):
    if not model or not scaler or not expected_cols:
        raise HTTPException(status_code=500, detail="ML model artifacts are not loaded properly.")

    try:
        # Build raw input dictionary matching Streamlit app.py format
        raw_input = {
            "Age": data.age,
            "RestingBP": data.resting_bp,
            "Cholesterol": data.cholesterol,
            "FastingBS": data.fasting_bs,
            "MaxHR": data.max_hr,
            "Oldpeak": data.oldpeak,
            "Sex_" + data.sex: 1,
            "ChestPainType_" + data.chest_pain: 1,
            "RestingECG_" + data.resting_ecg: 1,
            "ExerciseAngina_" + data.exercise_angina: 1,
            "ST_Slope_" + data.st_slope: 1,
        }

        input_df = pd.DataFrame([raw_input])

        # Fill missing expected columns with 0
        for col in expected_cols:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns to match model training setup
        input_df = input_df[expected_cols]

        # Scale numerical features
        numerical_cols = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        # Make prediction
        pred_val = int(model.predict(input_df)[0])
        
        # Calculate risk probability percentage
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]
            probability = round(float(proba) * 100, 1)
        else:
            probability = 95.0 if pred_val == 1 else 5.0

        risk_label = "High Risk of Heart Disease" if pred_val == 1 else "Low Risk of Heart Disease"
        risk_level = "HIGH" if pred_val == 1 else "LOW"

        # Generate clinical factor insights & recommendations
        key_factors = []
        recommendations = []

        if data.chest_pain == "ASY":
            key_factors.append("Asymptomatic Chest Pain presentation")
        elif data.chest_pain in ["ATA", "TA"]:
            key_factors.append(f"{data.chest_pain} Anginal Chest Pain reported")

        if data.resting_bp >= 140:
            key_factors.append(f"Elevated Resting Blood Pressure ({data.resting_bp} mm Hg)")
            recommendations.append("Monitor and manage blood pressure through dietary sodium reduction and medical follow-up.")

        if data.cholesterol >= 240:
            key_factors.append(f"High Cholesterol Level ({data.cholesterol} mg/dL)")
            recommendations.append("Consider lipid panel evaluation and cardiovascular risk management.")

        if data.fasting_bs == 1:
            key_factors.append("Fasting Blood Sugar > 120 mg/dL")
            recommendations.append("Evaluate glucose tolerance and glycemic control with primary healthcare provider.")

        if data.exercise_angina == "Y":
            key_factors.append("Exercise-Induced Angina detected")
            recommendations.append("Schedule a stress test and cardiology consultation.")

        if data.oldpeak > 1.5:
            key_factors.append(f"Significant ST Depression ({data.oldpeak})")

        if data.st_slope == "Flat" or data.st_slope == "Down":
            key_factors.append(f"Non-sloping/Down-sloping ST segment ({data.st_slope})")

        if not recommendations:
            if pred_val == 1:
                recommendations.append("Consult a cardiologist for a comprehensive cardiovascular assessment.")
            else:
                recommendations.append("Maintain a heart-healthy diet, regular aerobic exercise, and annual health checkups.")

        return {
            "prediction": pred_val,
            "risk_label": risk_label,
            "risk_level": risk_level,
            "probability": probability,
            "key_factors": key_factors,
            "recommendations": recommendations,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
