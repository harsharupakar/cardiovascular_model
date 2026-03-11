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
    df["bmi"]               = truncated_normal(26.5, 5.5, 16, 50, N).round(1)

    # Blood pressure correlated with BMI (r ≈ 0.35)
    noise_sbp = truncated_normal(0, 1, -3, 3, N)
    df["systolic_bp"]       = (118 + 14 * (0.35 * (df["bmi"] - 26.5) / 5.5 + 0.94 * noise_sbp)).clip(90, 180).round(1)

    noise_dbp = truncated_normal(0, 1, -3, 3, N)
    df["diastolic_bp"]      = (76 + 10 * (0.25 * (df["bmi"] - 26.5) / 5.5 + 0.97 * noise_dbp)).clip(60, 120).round(1)

    # Glucose correlated with BMI (r ≈ 0.30)
    noise_gluc = truncated_normal(0, 1, -3, 3, N)
    df["glucose"]           = (95 + 18 * (0.30 * (df["bmi"] - 26.5) / 5.5 + 0.95 * noise_gluc)).clip(70, 200).round(1)

    df["cholesterol_total"] = truncated_normal(185, 35, 120, 320, N).round(1)

    # Sleep inversely related to poor lifestyle
    noise_sleep = np.random.normal(0, 1, N)
    df["sleep_hours"]       = (7.0 + 1.3 * noise_sleep).clip(4, 10).round(1)

    # ── Ordinal ───────────────────────────────────────────────────────────────
    df["education"]           = np.random.choice([0,1,2,3], N, p=[0.05, 0.20, 0.45, 0.30])
    df["socioeconomic_status"]= np.random.choice([0,1,2],   N, p=[0.25, 0.50, 0.25])
    df["physical_activity"]   = np.random.choice([0,1,2,3], N, p=[0.40, 0.25, 0.20, 0.15])
    df["diet_quality"]        = np.random.choice([0,1,2],   N, p=[0.35, 0.40, 0.25])

    # ── Binary ────────────────────────────────────────────────────────────────
    df["smoking"]             = np.random.binomial(1, 0.12, N)
    # Alcohol slightly correlated with smoking
    df["alcohol_use"]         = (np.random.binomial(1, 0.35, N) | (df["smoking"] & np.random.binomial(1, 0.20, N))).clip(0, 1)
    df["pcos"]                = np.random.binomial(1, 0.10, N)
    df["family_history_cvd"]  = np.random.binomial(1, 0.18, N)

    # ── Pregnancy-gated ───────────────────────────────────────────────────────
    df["is_ever_pregnant"]        = np.random.binomial(1, 0.45, N)
    df["gestational_diabetes"]    = (df["is_ever_pregnant"] & np.random.binomial(1, 0.08, N)).clip(0, 1)
    df["preeclampsia"]            = (df["is_ever_pregnant"] & np.random.binomial(1, 0.06, N)).clip(0, 1)
    df["preterm_birth"]           = (df["is_ever_pregnant"] & np.random.binomial(1, 0.10, N)).clip(0, 1)

    # ── Risk Label ────────────────────────────────────────────────────────────
    high_risk = (
        (df["bmi"] >= 30) |
        (df["systolic_bp"] >= 140) |
        (df["glucose"] >= 126) |
        ((df["pcos"] == 1) & (df["bmi"] >= 28)) |
        (df["preeclampsia"] == 1)
    )
    moderate_risk = (
        ~high_risk &
        (df["bmi"] >= 25) &
        (df["smoking"] + df["alcohol_use"] + (df["physical_activity"] == 0).astype(int) + (df["sleep_hours"] < 6).astype(int) >= 2)
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
