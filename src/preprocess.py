"""
preprocess.py — Full preprocessing pipeline:
  - Reproductive feature gating (is_ever_pregnant)
  - Encoding (Standard, Ordinal, OneHot)
  - 6 interaction/engineered features
  - Feature correlation heatmap
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    set_seeds, ROOT_DIR, DATA_DIR, MODELS_DIR, OUTPUTS_DIR,
    CONTINUOUS_FEATURES, ORDINAL_FEATURES, BINARY_FEATURES,
    PREGNANCY_FEATURES, TARGET_COL, SEED
)
set_seeds()

# ─── Pregnancy Gate ───────────────────────────────────────────────────────────
def apply_pregnancy_gate(df: pd.DataFrame) -> pd.DataFrame:
    """Zero out pregnancy features for non-pregnant women and mark N/A."""
    df = df.copy()
    mask_not_pregnant = df["is_ever_pregnant"] == 0
    for feat in ["gestational_diabetes", "preeclampsia", "preterm_birth"]:
        df.loc[mask_not_pregnant, feat] = 0  # 0 = "not applicable" (never pregnant)
    return df

# ─── Interaction / Engineered Features ──────────────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bmi_bp"]          = df["bmi"] * df["systolic_bp"]
    df["glucose_bmi"]     = df["glucose"] * df["bmi"]
    df["pcos_bmi"]        = df["pcos"] * df["bmi"]
    df["age_activity"]    = df["age"] * df["physical_activity"]
    df["metabolic_index"] = df["bmi"] * df["glucose"]
    df["lifestyle_score"] = (
        df["smoking"].astype(int) +
        df["alcohol_use"].astype(int) +
        (df["physical_activity"] == 0).astype(int) +
        (df["sleep_hours"] < 6).astype(int)
    )
    return df

# ─── Correlation Heatmap ─────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, target="cvd_risk_binary"):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(16, 13))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.4, ax=ax,
        cbar_kws={"shrink": 0.8}, annot_kws={"size": 7}
    )
    ax.set_title("Feature Correlation Matrix — CVD Risk Dataset (Women 18–35)", fontsize=14, pad=16)
    plt.tight_layout()
    out_path = os.path.join(OUTPUTS_DIR, "feature_correlation_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

# ─── Main Preprocessing ───────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame, fit: bool = True, preprocessor=None):
    df = apply_pregnancy_gate(df)
    df = add_interaction_features(df)

    interaction_feats = ["bmi_bp","glucose_bmi","pcos_bmi","age_activity","metabolic_index","lifestyle_score"]
    all_binary        = BINARY_FEATURES + ["gestational_diabetes","preeclampsia","preterm_birth"]

    feature_cols = CONTINUOUS_FEATURES + interaction_feats + ORDINAL_FEATURES + all_binary
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df["cvd_risk_binary"].values if "cvd_risk_binary" in df.columns else None

    if fit:
        preprocessor = ColumnTransformer(transformers=[
            ("num",  StandardScaler(),  CONTINUOUS_FEATURES + interaction_feats),
            ("ord",  OrdinalEncoder(),  ORDINAL_FEATURES),
            ("bin",  "passthrough",     all_binary),
        ], remainder="drop")
        X_transformed = preprocessor.fit_transform(X)
        joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))
    else:
        X_transformed = preprocessor.transform(X)

    feature_names_out = (
        CONTINUOUS_FEATURES + interaction_feats +
        ORDINAL_FEATURES + all_binary
    )
    return X_transformed, y, preprocessor, feature_names_out

if __name__ == "__main__":
    raw_path = os.path.join(DATA_DIR, "raw_dataset.csv")
    if not os.path.exists(raw_path):
        print("raw_dataset.csv not found — run data/generate_dataset.py first")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    df_eng = apply_pregnancy_gate(df)
    df_eng = add_interaction_features(df_eng)
    plot_correlation_heatmap(df_eng)

    X, y, prep, feature_names = preprocess(df)
    print(f"X shape: {X.shape}  |  Feature count: {len(feature_names)}")
    print(f"High Risk samples: {y.sum()} / {len(y)} ({100*y.mean():.1f}%)")
    df_eng.to_csv(os.path.join(DATA_DIR, "engineered_dataset.csv"), index=False)
    print("Engineered dataset saved.")
