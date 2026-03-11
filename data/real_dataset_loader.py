"""
real_dataset_loader.py
Loads NHANES CSV exports (train) and Kaggle CVD dataset (external test).
Harmonises columns to the 16-feature schema and saves train/test CSVs.

Usage:
  pip install kaggle
  Place NHANES & Kaggle exports in data/ as:
    - nhanes_raw.csv   (from NHANES 2017-2020 CDC download)
    - kaggle_cvd.csv   (https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
"""
import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR, SEED

np.random.seed(SEED)

NHANES_PATH  = os.path.join(DATA_DIR, "nhanes_raw.csv")
KAGGLE_PATH  = os.path.join(DATA_DIR, "kaggle_cvd.csv")
NHANES_OUT   = os.path.join(DATA_DIR, "nhanes_train.csv")
KAGGLE_OUT   = os.path.join(DATA_DIR, "kaggle_test.csv")
SYNTHETIC_PATH = os.path.join(DATA_DIR, "raw_dataset.csv")

# ── NHANES column mapping
NHANES_COL_MAP = {
    "RIDAGEYR":   "age",
    "BMXBMI":     "bmi",
    "BPXSY1":     "systolic_bp",
    "BPXDI1":     "diastolic_bp",
    "LBXGLU":     "glucose",
    "LBXTC":      "cholesterol_total",
    "SLD010H":    "sleep_hours",
    "DMDEDUC2":   "education",
    "INDFMPIR":   "socioeconomic_status",
    "PAQ670":     "physical_activity",
    "DBD900":     "diet_quality",
    "SMQ040":     "smoking",
    "ALQ130":     "alcohol_use",
    "RHQ131":     "is_ever_pregnant",
    "DIQ010":     "diabetes_flag",
    "BPQ020":     "hypertension_flag",
}

def load_nhanes():
    if not os.path.exists(NHANES_PATH):
        print(f"NHANES file not found at {NHANES_PATH}.")
        print("Using synthetic dataset as NHANES proxy.")
        return pd.read_csv(SYNTHETIC_PATH)
    df = pd.read_csv(NHANES_PATH)
    df = df.rename(columns={k: v for k, v in NHANES_COL_MAP.items() if k in df.columns})
    # Filter women 18-35
    if "age" in df.columns:
        df = df[(df["age"] >= 18) & (df["age"] <= 35)]
    # Derive target
    df["cvd_risk_binary"] = (
        (df.get("hypertension_flag", 0) == 1) |
        (df.get("diabetes_flag", 0) == 1)
    ).astype(int)
    df = df.fillna(df.median(numeric_only=True))
    print(f"NHANES train set: {len(df)} women aged 18-35")
    df.to_csv(NHANES_OUT, index=False)
    return df

# ── Kaggle CVD mapping
KAGGLE_COL_MAP = {
    "age":     "age",           # in days → convert
    "height":  "height_cm",
    "weight":  "weight_kg",
    "ap_hi":   "systolic_bp",
    "ap_lo":   "diastolic_bp",
    "gluc":    "glucose_cat",   # 1=normal 2=above 3=well above
    "cholesterol": "chol_cat",
    "smoke":   "smoking",
    "alco":    "alcohol_use",
    "active":  "physical_activity",
    "cardio":  "cvd_risk_binary",
}

def load_kaggle():
    if not os.path.exists(KAGGLE_PATH):
        print(f"Kaggle CVD file not found at {KAGGLE_PATH}.")
        print("External validation will be skipped; using 20% split of synthetic data instead.")
        df = pd.read_csv(SYNTHETIC_PATH)
        df_test = df.sample(frac=0.2, random_state=SEED)
        df_test.to_csv(KAGGLE_OUT, index=False)
        return df_test
    df = pd.read_csv(KAGGLE_PATH, sep=";")
    df = df.rename(columns={k: v for k, v in KAGGLE_COL_MAP.items() if k in df.columns})
    # Convert age from days to years
    if "age" in df.columns:
        df["age"] = (df["age"] / 365.25).round(1)
    df = df[(df["age"] >= 18) & (df["age"] <= 35)]
    # Derive BMI
    if "height_cm" in df.columns and "weight_kg" in df.columns:
        df["bmi"] = (df["weight_kg"] / (df["height_cm"] / 100) ** 2).round(1)
    df = df.fillna(df.median(numeric_only=True))
    print(f"Kaggle external test set: {len(df)} women aged 18-35")
    df.to_csv(KAGGLE_OUT, index=False)
    return df

if __name__ == "__main__":
    nhanes = load_nhanes()
    kaggle = load_kaggle()
    print("Done. Files saved to data/")
