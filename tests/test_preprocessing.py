import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import apply_pregnancy_gate, add_interaction_features
from src.utils import CONTINUOUS_FEATURES, INTERACTION_FEATURES

def test_pregnancy_gate():
    df = pd.DataFrame({
        "is_ever_pregnant": [1, 0, 1],
        "gestational_diabetes": [1, 1, 0],
        "preeclampsia": [0, 1, 1],
        "preterm_birth": [1, 1, 0]
    })
    
    out_df = apply_pregnancy_gate(df)
    
    # Non-pregnant woman (idx 1) should have all 0s for pregnancy complications
    assert out_df.loc[1, "gestational_diabetes"] == 0
    assert out_df.loc[1, "preeclampsia"] == 0
    assert out_df.loc[1, "preterm_birth"] == 0
    
    # Pregnant woman (idx 0) should retain her 1
    assert out_df.loc[0, "gestational_diabetes"] == 1

def test_interaction_features():
    df = pd.DataFrame({
        "BMI": [25.0, 30.0],
        "blood_pressure": [120.0, 140.0],
        "glucose": [90.0, 110.0],
        "PCOS": [0, 1],
        "age": [25, 30],
        "activity": [3.0, 1.0]
    })
    
    out_df = add_interaction_features(df)
    
    assert "BMI_x_BP" in out_df.columns
    assert "Glucose_x_BMI" in out_df.columns
    assert "PCOS_x_BMI" in out_df.columns
    assert "Age_x_PhysicalActivity" in out_df.columns
    
    assert out_df.loc[0, "BMI_x_BP"] == 25.0 * 120.0
    assert out_df.loc[1, "PCOS_x_BMI"] == 1 * 30.0
    assert out_df.loc[0, "Age_x_PhysicalActivity"] == 25 * 3.0
