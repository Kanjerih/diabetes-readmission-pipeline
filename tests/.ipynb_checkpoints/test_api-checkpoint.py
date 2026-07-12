import pytest
from fastapi.testclient import TestClient
from src.app import app

# Initialize the test client with our FastAPI app instance
client = TestClient(app)

@pytest.fixture
def valid_high_risk_patient():
    """Fixture providing a standard valid payload matching the Pydantic schema."""
    return {
        "gender": 1.0,
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "admission_source_id": 7,
        "time_in_hospital": 6,
        "num_lab_procedures": 55,
        "num_procedures": 2,
        "num_medications": 18,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 3,
        "number_diagnoses": 9,
        "age_num": 65.0,
        "features": {
            "metformin": 1,
            "insulin": 1,
            "diabetesMed": 1,
            "diag_1_group_Diabetes": 1,
            "diag_2_group_Circulatory": 1
        }
    }

def test_health_endpoint():
    """Verify the liveness probe works and indicates the model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_successful_prediction(valid_high_risk_patient):
    """Verify that a valid payload returns the correct structural keys and boundaries."""
    response = client.post("/predict", json=valid_high_risk_patient)
    assert response.status_code == 200
    
    data = response.json()
    assert "readmission_probability" in data
    assert "decisions" in data
    
    # Assert probability bounds
    assert 0.0 <= data["readmission_probability"] <= 1.0
    
    # Assert all operational thresholds exist in the response
    decisions = data["decisions"]
    assert "threshold_0_45" in decisions
    assert "threshold_0_50" in decisions
    assert "threshold_0_55" in decisions
    
    # Assert decision outputs are strictly binary flags (0 or 1)
    assert decisions["threshold_0_45"] in [0, 1]

def test_invalid_payload_validation():
    """Verify that bad data types correctly trigger Pydantic validation errors (422)."""
    bad_payload = {
        "gender": "Not A Number",  # Invalid type string instead of float
        "time_in_hospital": "long"
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Unprocessable Entity