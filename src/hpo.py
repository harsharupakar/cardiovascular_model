"""
hpo.py — Hyperparameter optimization using Optuna.
Tunes hidden_size, dropout, lr, and weight_decay.
Logs all trials to MLflow.
"""
import os
import sys
import optuna
import mlflow
import pandas as pd
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, MODELS_DIR, set_seeds, SEED
from preprocess import preprocess
from classifier import run_training

set_seeds()
mlflow.set_experiment("CVD_MLP_HPO")

def objective(trial):
    # Suggest hyperparameters
    params = {
        'hidden_size':  trial.suggest_int("hidden_size", 64, 256, step=64),
        'dropout':      trial.suggest_float("dropout", 0.1, 0.5),
        'lr':          trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }
    
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, "raw_dataset.csv"))
    # Pre-preprocess for speed in HPO (optional: can do inside CV but slower)
    X, y, _, _ = preprocess(df, fit=True)
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        mlflow.log_param("use_ctgan", "False") # CTGAN off during search for speed
        
        # Run 5-fold CV (without CTGAN for HPO speed)
        mean_auc, std_auc = run_training(X, y, params, use_ctgan=False, n_folds=5, epochs=30)
        
        mlflow.log_metric("mean_auc", mean_auc)
        mlflow.log_metric("std_auc", std_auc)
        
    return mean_auc

def run_hpo(n_trials=20):
    print(f"\n=== Starting HPO ({n_trials} trials) ===")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best params
    best_params_path = os.path.join(MODELS_DIR, "best_hpo_params.joblib")
    joblib.dump(trial.params, best_params_path)
    print(f"Best params saved to {best_params_path}")
    
    return trial.params

if __name__ == "__main__":
    run_hpo(n_trials=10) # Using 10 for demonstration speed
