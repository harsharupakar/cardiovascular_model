"""
main_pipeline.py — Master script to run either:
    - classic MLP pipeline
    - joint-fusion multimodal pipeline
"""
import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
        DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        SEED,
        set_seeds,
        CONTINUOUS_FEATURES,
        ORDINAL_FEATURES,
        BINARY_FEATURES,
        PREGNANCY_FEATURES,
)
from preprocess import preprocess
from classifier import run_training, CVDClassifier
from calibration import calibrate_model, evaluate_calibration, plot_calibration_curve
from evaluate import run_metrics, run_shap_analysis
from shap_stability import analyze_shap_stability
from fairness_audit import run_fairness_audit
from ctgan_augment import augment_fold # Fix if path is different, but here it's fine
from joint_fusion import JointFusionHFNet, train_joint_fusion, predict_joint_fusion

set_seeds()
device = "cuda" if torch.cuda.is_available() else "cpu"

def _ensure_structural_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure structural echo feature columns exist.
    If missing, derive clinically informed proxies from available tabular features.
    """
    df = df.copy()
    required_struct = ["LVEF", "LVEDD", "WallMotion", "MitralRegurgitation"]
    if all(col in df.columns for col in required_struct):
        return df

    bp = df["blood_pressure"].astype(float)
    bmi = df["BMI"].astype(float)
    glucose = df["glucose"].astype(float)
    htn = df.get("hypertension", pd.Series(0, index=df.index)).astype(float)
    smoking = df.get("smoking", pd.Series(0, index=df.index)).astype(float)

    lvef = 68.0 - 0.20 * (bp - 120.0) - 0.30 * (bmi - 25.0) - 0.08 * (glucose - 100.0) - 5.0 * htn - 3.0 * smoking
    lvef = np.clip(lvef, 25.0, 72.0)

    lvedd = 46.0 + 0.10 * (bmi - 25.0) + 0.07 * (bp - 120.0) + 3.5 * htn + 1.5 * smoking
    lvedd = np.clip(lvedd, 35.0, 70.0)

    wall_motion = np.where(lvef <= 40.0, 2, np.where(lvef <= 50.0, 1, 0))
    mitral_regurg = np.where(lvedd >= 60.0, 3, np.where(lvedd >= 55.0, 2, np.where(lvedd >= 50.0, 1, 0)))

    df["LVEF"] = lvef
    df["LVEDD"] = lvedd
    df["WallMotion"] = wall_motion.astype(int)
    df["MitralRegurgitation"] = mitral_regurg.astype(int)
    return df


def _run_classic_mlp_pipeline(df: pd.DataFrame, params: dict):
    X, y, preprocessor_obj, feature_names = preprocess(df, fit=True)

    print("\n--- Phase 2: Training Final Model (with CTGAN) ---")
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        stratify=df['cvd_risk_binary'],
        random_state=SEED
    )

    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]

    aug_train_df = augment_fold(train_df, n_synthetic=300, fold_idx=0)
    X_train, y_train, _, _ = preprocess(aug_train_df, fit=False, preprocessor=preprocessor_obj)
    X_test, y_test, _, _ = preprocess(test_df, fit=False, preprocessor=preprocessor_obj)

    model = CVDClassifier(X_train.shape[1], hidden_size=params['hidden_size'], dropout=params['dropout']).to(device)
    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / max(sum(y_train), 1)]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])

    from torch.utils.data import DataLoader, TensorDataset
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    val_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    best_loss = float('inf')
    best_state = None
    patience_counter = 0
    model_path = os.path.join(MODELS_DIR, "cvd_classifier.pt")

    for epoch in range(150):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict().copy()
            torch.save(best_state, model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Final model saved to {model_path} with best val_loss={best_loss:.4f}")

    print("\n--- Phase 3: Probability Calibration ---")
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    val_loader = DataLoader(test_ds, batch_size=32)
    temp = calibrate_model(model, val_loader, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_test).to(device))
        probs = torch.sigmoid(logits / temp).cpu().numpy().flatten()

    evaluate_calibration(y_test, probs)
    plot_calibration_curve(y_test, probs, "MLP+CTGAN", os.path.join(OUTPUTS_DIR, "calibration_curve.png"))

    print("\n--- Phase 4: Full Metrics & Confusion Matrix ---")
    run_metrics(y_test, probs, name="Elite_MLP_v1")
    run_shap_analysis(model, X_train, X_test, feature_names)
    run_fairness_audit(y_test, (probs >= 0.5).astype(int), test_df)


def _run_joint_fusion_pipeline(df: pd.DataFrame, params: dict):
    print("\n--- Joint Fusion Pipeline (Early Multimodal Fusion) ---")
    df = _ensure_structural_columns(df)

    tab_features = [
        c for c in (CONTINUOUS_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES + PREGNANCY_FEATURES)
        if c in df.columns
    ]
    struct_features = [c for c in ["LVEF", "LVEDD", "WallMotion", "MitralRegurgitation"] if c in df.columns]

    if len(struct_features) == 0:
        raise RuntimeError("No structural features available for joint fusion.")

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=0.2,
        stratify=df['cvd_risk_binary'],
        random_state=SEED
    )

    train_df = df.loc[train_idx].copy()
    test_df = df.loc[test_idx].copy()

    tab_scaler = StandardScaler()
    struct_scaler = StandardScaler()

    X_tab_train = tab_scaler.fit_transform(train_df[tab_features].astype(float).values)
    X_struct_train = struct_scaler.fit_transform(train_df[struct_features].astype(float).values)
    y_train = train_df["cvd_risk_binary"].astype(int).values

    X_tab_test = tab_scaler.transform(test_df[tab_features].astype(float).values)
    X_struct_test = struct_scaler.transform(test_df[struct_features].astype(float).values)
    y_test = test_df["cvd_risk_binary"].astype(int).values

    model = JointFusionHFNet(
        tab_dim=X_tab_train.shape[1],
        struct_dim=X_struct_train.shape[1],
        d_model=64,
        n_heads=4,
        hidden=params.get("hidden_size", 128),
        dropout=params.get("dropout", 0.3),
    )

    results = train_joint_fusion(
        model=model,
        x_tab_train=X_tab_train,
        x_struct_train=X_struct_train,
        y_train=y_train,
        x_tab_val=X_tab_test,
        x_struct_val=X_struct_test,
        y_val=y_test,
        struct_feature_names=struct_features,
        device=device,
        lr=params.get("lr", 1e-3),
        weight_decay=params.get("weight_decay", 1e-4),
        batch_size=32,
        epochs=120,
        patience=12,
        aux_weight=0.35,
    )
    print(f"Joint fusion best val AUC: {results['best_val_auc']:.4f}")

    pred = predict_joint_fusion(model, X_tab_test, X_struct_test, device=device)
    probs = pred["probability"]

    run_metrics(y_test, probs, name="JointFusion_HF_v1")
    run_fairness_audit(y_test, (probs >= 0.5).astype(int), test_df)

    model_path = os.path.join(MODELS_DIR, "cvd_joint_fusion.pt")
    artifacts_path = os.path.join(MODELS_DIR, "joint_fusion_artifacts.joblib")
    torch.save(model.state_dict(), model_path)
    joblib.dump({
        "tab_features": tab_features,
        "struct_features": struct_features,
        "tab_scaler": tab_scaler,
        "struct_scaler": struct_scaler,
    }, artifacts_path)
    print(f"Saved joint model to {model_path}")
    print(f"Saved joint artifacts to {artifacts_path}")


def main(model_type: str = "mlp"):
    print("\n" + "="*50)
    print("STARTING FINAL ELITE CVD PIPELINE")
    print("="*50)

    # ── 1. Load Data & Params ──
    raw_path = os.path.join(DATA_DIR, "raw_dataset.csv")
    df = pd.read_csv(raw_path)

    params_path = os.path.join(MODELS_DIR, "best_hpo_params.joblib")
    if os.path.exists(params_path):
        params = joblib.load(params_path)
        print(f"Loaded best params: {params}")
    else:
        params = {'hidden_size': 128, 'dropout': 0.3, 'lr': 1e-3, 'weight_decay': 1e-4}
        print("Using default params.")

    if model_type == "joint_fusion":
        _run_joint_fusion_pipeline(df, params)
    else:
        _run_classic_mlp_pipeline(df, params)

    print("\n" + "="*50)
    print("ELITE PIPELINE COMPLETE. READY FOR DASHBOARD.")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CVD pipeline")
    parser.add_argument("--model", choices=["mlp", "joint_fusion"], default="mlp")
    args = parser.parse_args()
    main(model_type=args.model)
