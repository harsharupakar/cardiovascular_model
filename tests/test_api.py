import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app, load_assets

client = TestClient(app)

def test_health_check_fail_if_no_model():
    # If load_assets is NOT called, model is None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "unhealthy", "message": "Model or preprocessor missing"}

def test_explain_endpoint_schema():
    # Even if model is not loaded, it should return 503 instead of a pydantic error 
    # if the payload is correct.
    payload = {
        "age": 28,
        "BMI": 26.5,
        "blood_pressure": 120,
        "glucose": 95.0,
        "activity": 3.0,
        "education": 2,
        "socioeconomic_status": 1,
        "smoking": 0,
        "PCOS": 0,
        "hypertension": 0,
        "is_ever_pregnant": 0
    }
    
    response = client.post("/explain", json=payload)
    # Since model is not loaded in this pure test state
    assert response.status_code == 503
    assert response.json()["detail"] == "Model not loaded."
    
def test_predict_schema_validation_error():
    # Missing required fields like BMI
    payload = {
        "age": 28
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
