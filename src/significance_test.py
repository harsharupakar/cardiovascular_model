"""
significance_test.py — Statistical significance testing for research claims.
Uses paired t-tests to compare CTGAN vs SMOTE recall across folds.
"""
from scipy.stats import ttest_rel
import pandas as pd
import numpy as np
import os

def run_significance_test(metrics_a, metrics_b, name_a="CTGAN", name_b="SMOTE", out_path=None):
    """
    metrics_a/b: lists of metric values across folds (e.g. recall scores)
    """
    print(f"\n=== Significance Test: {name_a} vs {name_b} ===")
    t_stat, p_val = ttest_rel(metrics_a, metrics_b)
    
    print(f"  T-statistic: {t_stat:.4f}")
    print(f"  P-value    : {p_val:.4f}")
    
    if p_val < 0.05:
        print(f"  ✅ Result is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print(f"  ❌ Result is NOT statistically significant")
        
    if out_path:
        res = pd.DataFrame([{
            "Comparison": f"{name_a} vs {name_b}",
            "T-statistic": t_stat,
            "P-value": p_val,
            "Significant": p_val < 0.05
        }])
        res.to_csv(out_path, index=False)
        
    return p_val
