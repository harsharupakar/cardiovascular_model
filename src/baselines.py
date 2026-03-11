"""
baselines.py — LR, RandomForest, XGBoost, LightGBM with StratifiedKFold k=5.
Logs all metrics to MLflow. Outputs outputs/baseline_comparison.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score, precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, OUTPUTS_DIR, SEED, set_seeds
from preprocess import preprocess

set_seeds()
mlflow.set_experiment("CVD_Baselines")

MODELS = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    "Random_Forest":       RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED),
    "XGBoost":             XGBClassifier(n_estimators=200, use_label_encoder=False,
                                          eval_metric="logloss", random_state=SEED, verbosity=0),
    "LightGBM":            LGBMClassifier(n_estimators=200, class_weight="balanced",
                                          random_state=SEED, verbose=-1),
}

def count_params(model) -> int:
    try:
        if hasattr(model, "coef_"):
            return model.coef_.size
        if hasattr(model, "n_features_in_"):
            return model.n_features_in_ * 10  # rough proxy
    except Exception:
        pass
    return -1

def run_baseline(name, model, X, y, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    aucs, recalls, f1s, precs = [], [], [], []
    with mlflow.start_run(run_name=name):
        mlflow.log_param("model", name)
        mlflow.log_param("n_folds", n_folds)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            m = type(model)(**model.get_params())
            m.fit(X[tr], y[tr])
            proba = m.predict_proba(X[te])[:, 1]
            pred  = (proba >= 0.5).astype(int)
            aucs.append(roc_auc_score(y[te], proba))
            recalls.append(recall_score(y[te], pred))
            f1s.append(f1_score(y[te], pred))
            precs.append(precision_score(y[te], pred, zero_division=0))

        mean_auc    = np.mean(aucs)
        mean_recall = np.mean(recalls)
        mean_f1     = np.mean(f1s)
        mean_prec   = np.mean(precs)
        mlflow.log_metric("roc_auc",   mean_auc)
        mlflow.log_metric("recall",    mean_recall)
        mlflow.log_metric("f1",        mean_f1)
        mlflow.log_metric("precision", mean_prec)
        print(f"  {name:<22} AUC={mean_auc:.3f} Recall={mean_recall:.3f} F1={mean_f1:.3f}")
        return {
            "Model": name,
            "ROC_AUC": round(mean_auc, 4),
            "Recall_High_Risk": round(mean_recall, 4),
            "F1": round(mean_f1, 4),
            "Precision": round(mean_prec, 4),
            "AUC_std": round(np.std(aucs), 4),
        }

def run_all_baselines():
    raw_path = os.path.join(DATA_DIR, "raw_dataset.csv")
    df = pd.read_csv(raw_path)
    X, y, _, _ = preprocess(df, fit=True)
    print("\n=== Baseline Models (5-Fold CV) ===")
    results = []
    for name, model in MODELS.items():
        results.append(run_baseline(name, model, X, y))

    df_results = pd.DataFrame(results)
    out_path = os.path.join(OUTPUTS_DIR, "baseline_comparison.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\nBaseline comparison saved to {out_path}")
    print(df_results.to_string(index=False))
    return df_results

if __name__ == "__main__":
    run_all_baselines()
