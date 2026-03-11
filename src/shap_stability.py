"""
shap_stability.py — Analyzes SHAP value consistency across CV folds.
Outputs mean ± std importance bar chart to outputs/shap_stability.png.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import shap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUTS_DIR, SEED, set_seeds

set_seeds()

def analyze_shap_stability(fold_shap_values, feature_names, out_dir=OUTPUTS_DIR):
    """
    fold_shap_values: list of (n_samples, n_features) arrays from each fold
    """
    print("\n=== Running SHAP Stability Analysis ===")
    
    # Compute mean absolute SHAP per feature for each fold
    fold_importances = []
    for f_val in fold_shap_values:
        mean_abs_f = np.mean(np.abs(f_val), axis=0)
        fold_importances.append(mean_abs_f)
    
    fold_importances = np.array(fold_importances) # (n_folds, n_features)
    
    mean_imp = np.mean(fold_importances, axis=0)
    std_imp  = np.std(fold_importances,  axis=0)
    
    # ── Plot ──
    df_plot = pd.DataFrame({
        'Feature': feature_names,
        'Mean_SHAP': mean_imp,
        'Std_SHAP': std_imp
    }).sort_values(by='Mean_SHAP', ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(df_plot['Feature'], df_plot['Mean_SHAP'], xerr=df_plot['Std_SHAP'], 
             color='skyblue', capsize=5)
    plt.xlabel("Mean |SHAP Value| (Across Folds)")
    plt.title("SHAP Feature Importance Stability (Mean ± Std)")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_stability.png"))
    plt.close()
    
    print(f"SHAP stability plot saved to {os.path.join(out_dir, 'shap_stability.png')}")
