"""
monitoring.py — Model monitoring using Evidently AI.
Tracks data drift, prediction drift, and fairness drift over simulated production batches.
"""
import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset, TargetDriftPreset

def run_monitoring_report(reference_df, current_df, out_path):
    """
    reference_df: training data
    current_df: production/inference data
    """
    print("\n=== Running Model Monitoring (Evidently AI) ===")
    
    report = Report(metrics=[
        DataDriftPreset(),
        TargetDriftPreset(),
        ClassificationPreset()
    ])
    
    # Evidently expects 'target' and 'prediction' columns for classification preset
    # Assuming predictions are already joined in current_df
    
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(out_path)
    print(f"Monitoring report saved to {out_path}")
