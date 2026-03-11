"""
smote_augment.py — SMOTE baseline augmentation for 3-way comparison.
Outputs: data/smote_augmented.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR, SEED, set_seeds

set_seeds()

def smote_augment(df: pd.DataFrame, target_col="cvd_risk_binary") -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in [target_col, "cvd_risk"]]
    X = df[feature_cols].values
    y = df[target_col].values

    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)

    df_res = pd.DataFrame(X_res, columns=feature_cols)
    df_res[target_col] = y_res
    return df_res

if __name__ == "__main__":
    raw_path = os.path.join(DATA_DIR, "raw_dataset.csv")
    df = pd.read_csv(raw_path)
    print(f"Before SMOTE: {df['cvd_risk_binary'].value_counts().to_dict()}")
    df_aug = smote_augment(df)
    print(f"After  SMOTE: {df_aug['cvd_risk_binary'].value_counts().to_dict()}")
    out = os.path.join(DATA_DIR, "smote_augmented.csv")
    df_aug.to_csv(out, index=False)
    print(f"Saved {out}")
