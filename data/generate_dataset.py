"""
generate_dataset.py
Generates a realistic synthetic base dataset of 1,000 women aged 18–35
calibrated to NHANES 2017-2020 distributions.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm

SEED = 42
np.random.seed(SEED)
N = 1000
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_dataset.csv")

def truncated_normal(mean, std, low, high, size):
    a, b = (low - mean) / std, (high - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def generate():
    df = pd.DataFrame()

    # ── Continuous ────────────────────────────────────────────────────────────
    df["age"]               = truncated_normal(26.5, 5.0, 18, 35, N).round(1)
    df["BMI"]               = truncated_normal(26.5, 5.5, 16, 50, N).round(1)

    # Blood pressure correlated with BMI
    noise_bp = truncated_normal(0, 1, -3, 3, N)
    df["blood_pressure"]    = (118 + 14 * (0.35 * (df["BMI"] - 26.5) / 5.5 + 0.94 * noise_bp)).clip(90, 180).round(1)

    # Glucose correlated with BMI
    noise_gluc = truncated_normal(0, 1, -3, 3, N)
    df["glucose"]           = (95 + 18 * (0.30 * (df["BMI"] - 26.5) / 5.5 + 0.95 * noise_gluc)).clip(70, 200).round(1)

    # Activity as continuous (e.g., hours per week)
    df["activity"]          = truncated_normal(3.0, 2.0, 0, 14, N).round(1)

    # Cholesterol (mg/dL) — correlated with BMI
    noise_chol = truncated_normal(0, 1, -3, 3, N)
    df["cholesterol"]       = (175 + 20 * (0.25 * (df["BMI"] - 26.5) / 5.5 + 0.97 * noise_chol)).clip(100, 320).round(1)

    # Sleep duration (hours/night)
    df["sleep_duration"]    = truncated_normal(6.8, 1.2, 4, 10, N).round(1)

    # Alcohol (drinks per week)
    df["alcohol"]           = truncated_normal(2.5, 3.0, 0, 20, N).round(1)
    df["education"]           = np.random.choice([0,1,2,3], N, p=[0.05, 0.20, 0.45, 0.30])
    df["socioeconomic_status"]= np.random.choice([0,1,2],   N, p=[0.25, 0.50, 0.25])
    # diet_pattern: 0=poor, 1=moderate, 2=good
    df["diet_pattern"]        = np.random.choice([0,1,2],   N, p=[0.30, 0.45, 0.25])
    # stress_level: 0=low, 1=moderate, 2=high  (psychosocial risk)
    df["stress_level"]        = np.random.choice([0,1,2],   N, p=[0.25, 0.45, 0.30])

    # ── Binary ────────────────────────────────────────────────────────────────
    df["smoking"]             = np.random.binomial(1, 0.12, N)
    df["PCOS"]                = np.random.binomial(1, 0.10, N)
    df["hypertension"]        = (df["blood_pressure"] >= 140).astype(int)

    # ── Pregnancy-gated ───────────────────────────────────────────────────────
    df["is_ever_pregnant"]        = np.random.binomial(1, 0.45, N)
    df["gestational_diabetes"]    = (df["is_ever_pregnant"] & np.random.binomial(1, 0.08, N)).clip(0, 1)
    df["preeclampsia"]            = (df["is_ever_pregnant"] & np.random.binomial(1, 0.06, N)).clip(0, 1)
    df["preterm_birth"]           = (df["is_ever_pregnant"] & np.random.binomial(1, 0.10, N)).clip(0, 1)

    # ── Risk Label ────────────────────────────────────────────────────────────
    high_risk = (
        (df["BMI"] >= 30) |
        (df["blood_pressure"] >= 140) |
        (df["glucose"] >= 126) |
        (df["cholesterol"] >= 240) |
        ((df["PCOS"] == 1) & (df["BMI"] >= 28)) |
        (df["preeclampsia"] == 1)
    )
    moderate_risk = (
        ~high_risk &
        (
            (df["BMI"] >= 25) |
            (df["smoking"] == 1) |
            (df["stress_level"] == 2) |
            (df["sleep_duration"] < 5) |
            (df["alcohol"] >= 14)
        )
    )
    df["cvd_risk"] = 0  # Low by default
    df.loc[moderate_risk, "cvd_risk"] = 1
    df.loc[high_risk,     "cvd_risk"] = 2

    # Re-encode for binary classification (High=1, Not High=0)
    df["cvd_risk_binary"] = (df["cvd_risk"] == 2).astype(int)

    print(f"Dataset shape: {df.shape}")
    print(f"Risk distribution:\n{df['cvd_risk'].value_counts().sort_index()}")
    print(f"  0=Low: {(df['cvd_risk']==0).sum()} | 1=Moderate: {(df['cvd_risk']==1).sum()} | 2=High: {(df['cvd_risk']==2).sum()}")
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved to {OUT_PATH}")
    return df

if __name__ == "__main__":
    generate()
