import os
import json
import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import shap
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor

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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Diabetes Readmission Prediction API is active."}

# Define the expected absolute path to the saved artifact
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "diabetes_readmission_xgb_model.json")
BASELINE_PATH = os.path.join(BASE_DIR, "..", "models", "feature_baseline.json")

print("MODEL PATH:", MODEL_PATH)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

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

# Load the training-data baseline statistics used for drift comparison
if os.path.exists(BASELINE_PATH):
    with open(BASELINE_PATH, "r") as f:
        feature_baseline = json.load(f)
else:
    feature_baseline = {}

# The flat, always-present numeric fields we log and monitor for drift.
# (The free-form 'features' dict is excluded since its keys vary request to request.)
MONITORED_FIELDS = [
    "gender", "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses", "age_num"
]

# ==========================================
# DATABASE: connection + table setup for prediction logging
# ==========================================
DATABASE_URL = os.environ.get("DATABASE_URL")


def get_db_connection():
    """Opens a new database connection. Returns None if DATABASE_URL isn't configured
    or the connection fails -- callers must handle that gracefully, since prediction
    serving should never be blocked by a monitoring/logging failure."""
    if not DATABASE_URL:
        return None
    try:
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return None


def init_db():
    """Creates the prediction_logs table if it doesn't already exist. Safe to call on
    every startup. Failures here are logged but never crash the app -- monitoring is
    an add-on, not a hard dependency for serving predictions."""
    conn = get_db_connection()
    if conn is None:
        logger.warning("DATABASE_URL not set or unreachable -- prediction logging is disabled.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    gender FLOAT,
                    admission_type_id INT,
                    discharge_disposition_id INT,
                    admission_source_id INT,
                    time_in_hospital INT,
                    num_lab_procedures INT,
                    num_procedures INT,
                    num_medications INT,
                    number_outpatient INT,
                    number_emergency INT,
                    number_inpatient INT,
                    number_diagnoses INT,
                    age_num FLOAT,
                    readmission_probability FLOAT
                )
            """)
        conn.commit()
        logger.info("Database table 'prediction_logs' ready.")
    except Exception as e:
        logger.error(f"Failed to initialize database table: {str(e)}")
    finally:
        conn.close()


def log_prediction(input_dict, probability):
    """Best-effort insert of a single prediction into the logs table. Never raises --
    a monitoring write failure should not affect the actual prediction response."""
    conn = get_db_connection()
    if conn is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO prediction_logs
                    (gender, admission_type_id, discharge_disposition_id, admission_source_id,
                     time_in_hospital, num_lab_procedures, num_procedures, num_medications,
                     number_outpatient, number_emergency, number_inpatient, number_diagnoses,
                     age_num, readmission_probability)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    input_dict.get("gender"), input_dict.get("admission_type_id"),
                    input_dict.get("discharge_disposition_id"), input_dict.get("admission_source_id"),
                    input_dict.get("time_in_hospital"), input_dict.get("num_lab_procedures"),
                    input_dict.get("num_procedures"), input_dict.get("num_medications"),
                    input_dict.get("number_outpatient"), input_dict.get("number_emergency"),
                    input_dict.get("number_inpatient"), input_dict.get("number_diagnoses"),
                    input_dict.get("age_num"), probability
                )
            )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log prediction to database: {str(e)}")
    finally:
        conn.close()


# Initialize the database table at startup (safe no-op if DATABASE_URL isn't set)
init_db()


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
        # shap_values can be a list (per-class) or a single array depending on SHAP version;
        # normalize to the row-level contribution array before zipping
        if isinstance(shap_values, list):
            row_shap = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            row_shap = shap_values[0]
        # Cast numpy floats to native Python floats -- required for JSON serialization,
        # otherwise FastAPI throws an unhandled TypeError during response encoding
        # (this happens AFTER the try block returns, which is why it surfaced as a
        # bare 500 "Internal Server Error" instead of the custom 400 message below)
        feature_contributions = {
            k: float(v) for k, v in zip(expected_features, row_shap)
        }
        
        # --- LOG THE RESULT ---
        logger.info(f"Prediction result: {probability}")

        # --- LOG TO DATABASE FOR MONITORING (best-effort, never blocks the response) ---
        log_prediction(input_dict, probability)
        
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


@app.get("/monitoring/drift")
def monitoring_drift(recent_n: int = 200, drift_threshold: float = 2.0):
    """Compares the distribution of recently logged prediction requests against the
    training-data baseline for each monitored feature, using a standardized mean
    shift: abs(recent_mean - baseline_mean) / baseline_std. A feature is flagged as
    'drifted' when this exceeds drift_threshold (default: 2 standard deviations).

    This is a lightweight, explainable drift signal -- not a substitute for more
    rigorous methods like Population Stability Index or KS-tests, but a reasonable
    first line of monitoring without additional infrastructure.
    """
    if not feature_baseline:
        raise HTTPException(status_code=503, detail="No training baseline available. Retrain the model to generate models/feature_baseline.json.")

    conn = get_db_connection()
    if conn is None:
        raise HTTPException(status_code=503, detail="Database unavailable. Check that DATABASE_URL is configured correctly.")

    try:
        with conn.cursor() as cur:
            columns = ", ".join(MONITORED_FIELDS)
            cur.execute(
                f"SELECT {columns} FROM prediction_logs ORDER BY created_at DESC LIMIT %s",
                (recent_n,)
            )
            rows = cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query prediction logs: {str(e)}")
    finally:
        conn.close()

    if not rows:
        return {
            "status": "no_data",
            "message": "No logged predictions yet. Drift cannot be assessed until requests have been served.",
            "sample_size": 0
        }

    recent_df = pd.DataFrame(rows)

    report = {}
    drifted_features = []
    for feature in MONITORED_FIELDS:
        if feature not in feature_baseline or feature not in recent_df.columns:
            continue

        baseline_mean = feature_baseline[feature]["mean"]
        baseline_std = feature_baseline[feature]["std"]
        recent_values = recent_df[feature].dropna()

        if len(recent_values) == 0 or baseline_std == 0:
            continue

        recent_mean = float(recent_values.mean())
        drift_score = abs(recent_mean - baseline_mean) / baseline_std
        is_drifted = drift_score > drift_threshold

        if is_drifted:
            drifted_features.append(feature)

        report[feature] = {
            "baseline_mean": round(baseline_mean, 4),
            "recent_mean": round(recent_mean, 4),
            "drift_score": round(drift_score, 4),
            "drifted": is_drifted
        }

    return {
        "status": "drift_detected" if drifted_features else "stable",
        "sample_size": len(rows),
        "drift_threshold": drift_threshold,
        "drifted_features": drifted_features,
        "feature_report": report
    }


print("deploy refresh")
