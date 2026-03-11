"""
evaluate.py — Comprehensive model evaluation:
  - Classification report (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Confusion Matrix
  - ROC Curve
  - SHAP Explainability (Beeswarm, Waterfall, Force, Dependence)
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import shap
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, 
    auc, precision_recall_curve, average_precision_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import OUTPUTS_DIR, SEED, set_seeds

set_seeds()

def run_metrics(y_true, y_prob, name="Model", out_dir=OUTPUTS_DIR):
    y_pred = (y_prob >= 0.5).astype(int)
    
    print(f"\n=== Evaluation: {name} ===")
    print(classification_report(y_true, y_pred))
    
    # ── ROC Curve ──
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic — {name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()
    
    # ── Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix — {name}')
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
    plt.close()
    
    return roc_auc

def run_shap_analysis(model, X_train, X_test, feature_names, out_dir=OUTPUTS_DIR):
    """
    Using shap.KernelExplainer for PyTorch model robustness.
    """
    print("\n=== Running SHAP Explainability Analysis ===")
    
    # Use a small background dataset for speed in KernelExplainer
    background = shap.sample(X_train, 50)
    
    def predict_fn(x):
        x_tensor = torch.FloatTensor(x)
        with torch.no_grad():
            outputs = model(x_tensor)
            return torch.sigmoid(outputs).cpu().numpy().flatten()
    
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test[:100]) # Explain first 100 test samples
    
    # ── Beeswarm Plot ──
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
    plt.title("Global Feature Importance (SHAP Beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"))
    plt.close()
    
    # ── Waterfall Plot (for 1st test sample) ──
    plt.figure(figsize=(10, 6))
    # Note: waterfall expects an Explanation object
    exp = shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, 
                          data=X_test[0], feature_names=feature_names)
    shap.plots.waterfall(exp, show=False)
    plt.title("Local Explanation — High Risk Patient (SHAP Waterfall)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_waterfall.png"))
    plt.close()
    
    # ── Dependence Plot (Age vs Risk) ──
    if "age" in feature_names:
        age_idx = list(feature_names).index("age")
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(age_idx, shap_values, X_test[:100], feature_names=feature_names, show=False)
        plt.title("SHAP Dependence Plot — Age vs Risk")
        plt.savefig(os.path.join(out_dir, "shap_dependence_age.png"))
        plt.close()
    
    print("SHAP plots saved to outputs/")
