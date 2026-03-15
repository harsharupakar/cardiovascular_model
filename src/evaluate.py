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
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"\n=== Evaluation: {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
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

def run_shap_analysis(model, X_train, X_test, feature_names):
    """
    Using shap.DeepExplainer with fallback to shap.KernelExplainer.
    Saves outputs to outputs/shap/
    """
    out_dir = os.path.join(OUTPUTS_DIR, "shap")
    os.makedirs(out_dir, exist_ok=True)
    
    print("\n=== Running SHAP Explainability Analysis ===")
    
    X_train_t = torch.FloatTensor(X_train)
    X_test_t = torch.FloatTensor(X_test[:100]) # Explain first 100 test samples
    
    try:
        explainer = shap.DeepExplainer(model, X_train_t[:100])
        shap_values = explainer.shap_values(X_test_t)
        # DeepExplainer returns [num_samples, num_features] for scalar output
    except Exception as e:
        print(f"DeepExplainer failed: {e}. Falling back to KernelExplainer.")
        background = shap.sample(X_train, 50)
        def predict_fn(x):
            x_tensor = torch.FloatTensor(x)
            with torch.no_grad():
                return torch.sigmoid(model(x_tensor)).cpu().numpy().flatten()
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_test[:100])
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # ── Top Factors per patient ──
    # Save a list of top 3 features for each patient in the explained set
    top_features = []
    for i in range(len(shap_values)):
        sample_shap = np.abs(shap_values[i])
        top_idx = np.argsort(sample_shap)[-3:][::-1] # indices of top 3
        top_features.append(", ".join([feature_names[int(j)] for j in top_idx]))
    
    pd.DataFrame({
        'Patient_Index': range(len(top_features)),
        'Top_3_SHAP_Factors': top_features
    }).to_csv(os.path.join(out_dir, "top_shap_factors.csv"), index=False)
    
    # ── Beeswarm Plot ──
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test[:100], feature_names=feature_names, show=False)
    plt.title("Global Feature Importance (SHAP Beeswarm)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_beeswarm.png"))
    plt.close()
    
    # ── Waterfall Plot (for 1st test sample) ──
    plt.figure(figsize=(10, 6))
    if hasattr(explainer, "expected_value"):
        base_val = explainer.expected_value
        if isinstance(base_val, list) or isinstance(base_val, np.ndarray):
            base_val = base_val[0]
    else:
        base_val = 0.5 # fallback dummy
        
    exp = shap.Explanation(values=np.squeeze(shap_values[0]), base_values=base_val, 
                          data=np.squeeze(X_test[0]), feature_names=feature_names)
    shap.plots.waterfall(exp, show=False)
    plt.title("Local Explanation (SHAP Waterfall)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_waterfall.png"))
    plt.close()
    
    print(f"SHAP plots and top factors saved to {out_dir}")
