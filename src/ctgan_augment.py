"""
ctgan_augment.py — Per-fold CTGAN augmentation (no data leakage) + SDV quality check.
The CTGAN is fitted ONLY on train fold data to prevent test contamination.
"""
import os
import sys
import numpy as np
import pandas as pd
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, MODELS_DIR, OUTPUTS_DIR, SEED, set_seeds, CONTINUOUS_FEATURES, ORDINAL_FEATURES, BINARY_FEATURES, PREGNANCY_FEATURES
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

def check_fidelity_return_report(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, fold_idx: int, model_name: str):
    """Run SDV evaluate_quality — checks Column Shapes, Column Pair Trends, Correlation."""
    try:
        metadata = build_metadata(real_df)
        common_cols = [c for c in real_df.columns if c in synthetic_df.columns]
        report = evaluate_quality(
            real_data=real_df[common_cols],
            synthetic_data=synthetic_df[common_cols],
            metadata=metadata,
        )
        score = report.get_score()
        print(f"  Fold {fold_idx} {model_name} SDV Quality Score: {score:.4f}", end="")
        if score >= SDV_QUALITY_THRESHOLD:
            print(" [PASSED]")
        else:
            print(f" [FAILED] (THRESHOLD: {SDV_QUALITY_THRESHOLD})")
        return score, report
    except Exception as e:
        print(f"  Fold {fold_idx} SDV check error: {e}")
        return 0.0, None

def augment_fold(X_train_df: pd.DataFrame, n_synthetic: int = 300,
                 fold_idx: int = 0, epochs: int = 200) -> pd.DataFrame:
    """
    Full augmentation for one fold:
    1. Train CTGAN on training fold only (High Risk)
    2. Try TVAE fallback if CTGAN fidelity < threshold
    3. Generate synthetic High Risk samples
    4. Save model, datasets, and fidelity report
    """
    print(f"\nFold {fold_idx}: Training CTGAN (epochs={epochs})...")
    high_risk_df = X_train_df[X_train_df["cvd_risk_binary"] == 1].reset_index(drop=True)

    if len(high_risk_df) < 10:
        print(f"  Fold {fold_idx}: Too few High Risk samples ({len(high_risk_df)}), skipping CTGAN.")
        return X_train_df

    metadata = build_metadata(high_risk_df)
    
    synthesizer_ctgan = CTGANSynthesizer(metadata, epochs=epochs, verbose=False, cuda=False)
    synthesizer_ctgan.fit(high_risk_df)
    synthetic_ctgan = synthesizer_ctgan.sample(num_rows=n_synthetic)
    synthetic_ctgan["cvd_risk_binary"] = 1

    score_ctgan, report_ctgan = check_fidelity_return_report(high_risk_df, synthetic_ctgan, fold_idx, "CTGAN")
    
    best_synthetic = synthetic_ctgan
    best_score = score_ctgan
    best_model = synthesizer_ctgan
    best_report = report_ctgan
    best_name = "CTGAN"

    if score_ctgan < SDV_QUALITY_THRESHOLD:
        print(f"  Fold {fold_idx}: CTGAN fidelity low. Falling back to TVAE...")
        synthesizer_tvae = TVAESynthesizer(metadata, epochs=epochs, cuda=False)
        synthesizer_tvae.fit(high_risk_df)
        synthetic_tvae = synthesizer_tvae.sample(num_rows=n_synthetic)
        synthetic_tvae["cvd_risk_binary"] = 1
        
        score_tvae, report_tvae = check_fidelity_return_report(high_risk_df, synthetic_tvae, fold_idx, "TVAE")
        
        if score_tvae > score_ctgan:
            best_synthetic = synthetic_tvae
            best_score = score_tvae
            best_model = synthesizer_tvae
            best_report = report_tvae
            best_name = "TVAE"

    if best_score < SDV_QUALITY_THRESHOLD:
        print(f"  Fold {fold_idx}: Both models failed quality threshold. Using real data only.")
        return X_train_df

    # Save artifacts
    fidelity_dir = os.path.join(OUTPUTS_DIR, "fidelity")
    os.makedirs(fidelity_dir, exist_ok=True)
    
    model_path = os.path.join(MODELS_DIR, f"{best_name.lower()}_model_fold{fold_idx}.pkl")
    best_model.save(model_path)
    
    dataset_path = os.path.join(fidelity_dir, f"augmented_dataset_fold{fold_idx}.csv")
    best_synthetic.to_csv(dataset_path, index=False)
    
    if best_report is not None:
        report_path = os.path.join(fidelity_dir, f"fidelity_report_fold{fold_idx}.csv")
        try:
            details = best_report.get_details(property_name='Column Shapes')
            details.to_csv(report_path, index=False)
        except Exception as e:
            print(f"  Fold {fold_idx} could not save fidelity report details: {e}")

    augmented = pd.concat([X_train_df, best_synthetic], ignore_index=True)
    print(f"  Fold {fold_idx}: Augmented set size: {len(augmented)} ({len(best_synthetic)} synthetic added using {best_name})")
    return augmented

if __name__ == "__main__":
    print("CTGAN augment module — imported by classifier.py for per-fold augmentation.")
    print("Run classifier.py to execute full StratifiedKFold + CTGAN pipeline.")
