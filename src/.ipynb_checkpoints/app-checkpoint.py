import os
import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import shap
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Readmission Prediction Service",
    description="Production API serving an optimized XGBoost classifier for minority-class readmission risk.",
    version="1.0.0"
)


# ADD THE MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-app.onrender.com"],  
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Diabetes Readmission Prediction API is active."}

# Define the expected absolute path to the saved artifact
MODEL_PATH = os.path.join("models", "diabetes_readmission_xgb_model.json")

# Load model artifact safely at runtime startup
if os.path.exists(MODEL_PATH):
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    # Extract exact features expected by the model to prevent array shape alignment errors
    expected_features = model.get_booster().feature_names
    # Initialize the SHAP explainer once at startup
    explainer = shap.TreeExplainer(model)
else:
    model = None
    explainer = None
    expected_features = []

class PatientData(BaseModel):
    """Pydantic schema representing required features matching the model definition."""
    gender: float = Field(..., description="0 for Female, 1 for Male")
    admission_type_id: int
    discharge_disposition_id: int
    admission_source_id: int
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    number_diagnoses: int
    age_num: float
    # Include default placeholders for sparse categorical dummy arrays
    # (Any features omitted during raw input will be imputed to 0 downstream)
    features: dict = Field(default_factory=dict, description="Dictionary containing remaining dummy columns or medication flags")

@app.get("/health")
def health_check():
    """Liveness/Readiness probe endpoint for container orchestration platforms."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model artifact is missing or unreadable.")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict")
def predict_readmission(patient: PatientData):
    """Generates readmission probabilities and applies custom operational decision thresholds."""
    if model is None or explainer is None: 
        raise HTTPException(status_code=500, detail="Prediction engine or explainer is uninitialized.")
    
    # --- LOG THE REQUEST ---
    logger.info(f"Incoming prediction request: {patient.model_dump()}")
    
    try:
        # Convert Pydantic object properties into a base dictionary layout
        input_dict = {k: v for k, v in patient.model_dump().items() if k != 'features'}
        # Unpack remaining optional sub-features directly into the primary dictionary layer
        input_dict.update(patient.features)
        
        # Build the structured payload dataframe matching our training scheme
        input_df = pd.DataFrame([input_dict])
        
        # Reindex data arrays to enforce exact alignment with training feature columns
        # Missing dummy variations are filled automatically with 0
        final_df = input_df.reindex(columns=expected_features, fill_value=0)
        
        # Calculate raw operational predictive probabilities
        probability = float(model.predict_proba(final_df)[:, 1][0])


        # Calculate SHAP contributions
        shap_values = explainer.shap_values(final_df)
        feature_contributions = dict(zip(expected_features, shap_values[0]))
        
        # --- LOG THE RESULT ---
        logger.info(f"Prediction result: {probability}")
        
        # Evaluate operational outcome tags across our verified threshold baselines
        return {
            "readmission_probability": round(probability, 4),
            "feature_contributions": feature_contributions,
            "decisions": {
                "threshold_0_45": int(probability > 0.45), # Catch-all clinical setting (Recall: 70%)
                "threshold_0_50": int(probability > 0.50), # Balanced clinical baseline (Recall: 58%)
                "threshold_0_55": int(probability > 0.55)  # Resource-conservative setting (Precision: 21%)
            }
        }
    except Exception as e:
        # --- LOG THE ERROR ---
        logger.error(f"Inference pipeline execution error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Inference pipeline execution error: {str(e)}")