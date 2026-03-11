"""
external_validation.py — Validates the model on the unseen Kaggle CVD dataset.
Calculates ROC-AUC, Recall, and Brier Score on external data.
"""
import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.metrics import roc_auc_score, recall_score, brier_score_loss

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, MODELS_DIR, OUTPUTS_DIR
from preprocess import preprocess

def run_external_validation(model, device="cpu"):
    print("\n=== Running External Validation (Kaggle CVD Dataset) ===")
    
    kaggle_path = os.path.join(DATA_DIR, "kaggle_test.csv")
    if not os.path.exists(kaggle_path):
        print("Kaggle test set not found. Skipping external validation.")
        return
        
    df_kaggle = pd.read_csv(kaggle_path)
    
    # Preprocess Kaggle data using the fitted train preprocessor
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    X_ext, y_ext, _, _ = preprocess(df_kaggle, fit=False, preprocessor=preprocessor)
    
    X_ext_tensor = torch.FloatTensor(X_ext).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(X_ext_tensor)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)
        
    auc = roc_auc_score(y_ext, probs)
    recall = recall_score(y_ext, preds)
    brier = brier_score_loss(y_ext, probs)
    
    print(f"External ROC-AUC: {auc:.4f}")
    print(f"External Recall  : {recall:.4f}")
    print(f"External Brier   : {brier:.4f}")
    
    res_df = pd.DataFrame([{
        "Metric": "ROC-AUC", "Value": auc,
        "Metric": "Recall", "Value": recall,
        "Metric": "Brier", "Value": brier
    }])
    res_df.to_csv(os.path.join(OUTPUTS_DIR, "external_validation.csv"), index=False)
    print("External validation results saved.")
    
    return auc
