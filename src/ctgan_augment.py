"""
ctgan_augment.py — Per-fold CTGAN augmentation (no data leakage) + SDV quality check.
The CTGAN is fitted ONLY on train fold data to prevent test contamination.
"""
import os
import sys
import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, MODELS_DIR, SEED, set_seeds, CONTINUOUS_FEATURES, ORDINAL_FEATURES, BINARY_FEATURES, PREGNANCY_FEATURES
import joblib

set_seeds()

SDV_QUALITY_THRESHOLD = 0.80

def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    # Mark column types explicitly
    for col in CONTINUOUS_FEATURES:
        if col in df.columns:
            metadata.update_column(col, sdtype="numerical")
    for col in ORDINAL_FEATURES:
        if col in df.columns:
            metadata.update_column(col, sdtype="numerical")
    for col in BINARY_FEATURES + ["gestational_diabetes", "preeclampsia", "preterm_birth"]:
        if col in df.columns:
            metadata.update_column(col, sdtype="categorical")
    if "cvd_risk_binary" in df.columns:
        metadata.update_column("cvd_risk_binary", sdtype="categorical")
    return metadata

def train_ctgan_on_fold(X_fold_df: pd.DataFrame, epochs: int = 200) -> CTGANSynthesizer:
    """Train CTGAN on a single fold's training data. No leakage from test fold."""
    metadata = build_metadata(X_fold_df)
    synthesizer = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        verbose=False,
        cuda=False,
    )
    synthesizer.fit(X_fold_df)
    return synthesizer

def generate_synthetic(synthesizer: CTGANSynthesizer, n: int = 300) -> pd.DataFrame:
    return synthesizer.sample(num_rows=n)

def check_fidelity(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, fold_idx: int) -> float:
    """Run SDV evaluate_quality — checks Column Shapes, Column Pair Trends, Correlation."""
    try:
        metadata = build_metadata(real_df)
        # Align columns
        common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
        report = evaluate_quality(
            real_data=real_df[common_cols],
            synthetic_data=synthetic_df[common_cols],
            metadata=metadata,
        )
        score = report.get_score()
        print(f"  Fold {fold_idx} SDV Quality Score: {score:.4f}", end="")
        if score >= SDV_QUALITY_THRESHOLD:
            print(" [PASSED]")
        else:
            print(f" [FAILED] (THRESHOLD: {SDV_QUALITY_THRESHOLD})")
        return score
    except Exception as e:
        print(f"  Fold {fold_idx} SDV check error: {e}")
        return 0.0

def augment_fold(X_train_df: pd.DataFrame, n_synthetic: int = 300,
                 fold_idx: int = 0, epochs: int = 200) -> pd.DataFrame:
    """
    Full augmentation for one fold:
    1. Train CTGAN on training fold only
    2. Generate synthetic High Risk samples
    3. Run SDV fidelity check
    4. Return augmented training set
    """
    print(f"\nFold {fold_idx}: Training CTGAN (epochs={epochs})...")
    high_risk_df = X_train_df[X_train_df["cvd_risk_binary"] == 1].reset_index(drop=True)

    if len(high_risk_df) < 10:
        print(f"  Fold {fold_idx}: Too few High Risk samples ({len(high_risk_df)}), skipping CTGAN.")
        return X_train_df

    synthesizer = train_ctgan_on_fold(high_risk_df, epochs=epochs)
    synthetic   = generate_synthetic(synthesizer, n=n_synthetic)
    synthetic["cvd_risk_binary"] = 1

    score = check_fidelity(high_risk_df, synthetic, fold_idx)
    if score < SDV_QUALITY_THRESHOLD:
        print(f"  Fold {fold_idx}: Quality below threshold. Using real data only for this fold.")
        return X_train_df

    augmented = pd.concat([X_train_df, synthetic], ignore_index=True)
    print(f"  Fold {fold_idx}: Augmented set size: {len(augmented)} ({len(synthetic)} synthetic added)")
    return augmented

if __name__ == "__main__":
    print("CTGAN augment module — imported by classifier.py for per-fold augmentation.")
    print("Run classifier.py to execute full StratifiedKFold + CTGAN pipeline.")
