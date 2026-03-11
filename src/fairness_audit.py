"""
fairness_audit.py — Audit for bias across Socioeconomic Status and Age subgroups.
Uses Fairlearn to calculate Equal Opportunity and Demographic Parity differences.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equal_opportunity_difference
from sklearn.metrics import recall_score, precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUTS_DIR

def run_fairness_audit(y_true, y_pred, df_subgroups, out_dir=OUTPUTS_DIR):
    """
    df_subgroups: DataFrame containing sensitive features ('socioeconomic_status', 'age_bucket')
    """
    print("\n=== Running Fairness Audit ===")
    
    # ── 1. Audit by Socioeconomic Status ──
    metrics = {
        'Selection Rate': selection_rate,
        'Recall (Sensitivity)': recall_score,
        'Precision': precision_score
    }
    
    mf_ses = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=df_subgroups['socioeconomic_status']
    )
    
    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=df_subgroups['socioeconomic_status'])
    eo_diff = equal_opportunity_difference(y_true, y_pred, sensitive_features=df_subgroups['socioeconomic_status'])
    
    print("\nFairness Metrics (Socioeconomic Status):")
    print(mf_ses.by_group)
    print(f"Demographic Parity Difference: {dp_diff:.4f}")
    print(f"Equal Opportunity Difference: {eo_diff:.4f}")
    
    # ── 2. Audit by Age Bucket ──
    # Create age buckets if not present
    if 'age_bucket' not in df_subgroups.columns and 'age' in df_subgroups.columns:
        df_subgroups['age_bucket'] = pd.cut(df_subgroups['age'], bins=[18, 23, 29, 36], labels=['18-22', 23-28, '29-35'])
    
    mf_age = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=df_subgroups['age_bucket']
    )
    
    print("\nFairness Metrics (Age Bucket):")
    print(mf_age.by_group)
    
    # ── Save Results ──
    res_path = os.path.join(out_dir, "fairness_audit_results.csv")
    mf_ses.by_group.to_csv(res_path)
    print(f"Fairness audit saved to {res_path}")
    
    return eo_diff
