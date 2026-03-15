import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ctgan_augment import augment_fold

def test_ctgan_augmentation_fallback(monkeypatch):
    """
    Test that augmentation gracefully falls back or returns original data 
    if few High Risk samples exist.
    """
    # Create fake training data
    df = pd.DataFrame({
        "age": [25]*20,
        "BMI": [22]*20,
        "blood_pressure": [110]*20,
        "glucose": [90]*20,
        "activity": [5]*20,
        "education": [1]*20,
        "socioeconomic_status": [1]*20,
        "smoking": [0]*20,
        "PCOS": [0]*20,
        "hypertension": [0]*20,
        "is_ever_pregnant": [0]*20,
        "gestational_diabetes": [0]*20,
        "preeclampsia": [0]*20,
        "preterm_birth": [0]*20,
        "cvd_risk_binary": [0]*15 + [1]*5  # Only 5 high risk -> should skip CTGAN
    })
    
    # We expect it to skip CTGAN and return unmodified df because <10 High Risk samples
    out_df = augment_fold(df, n_synthetic=50, fold_idx=1)
    
    assert len(out_df) == 20
    assert (out_df["cvd_risk_binary"] == 1).sum() == 5
