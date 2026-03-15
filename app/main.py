"""
main.py — FastAPI backend providing CVD risk prediction, 
SHAP explanations, and model metadata.
"""
import os
import sys
import torch
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import json
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import DATA_DIR, MODELS_DIR, prob_to_label, CONTINUOUS_FEATURES, INTERACTION_FEATURES, ORDINAL_FEATURES, BINARY_FEATURES, PREGNANCY_FEATURES
from src.classifier import CVDClassifier
from src.joint_fusion import JointFusionHFNet
from src.uncertainty import mc_dropout_predict, get_risk_with_confidence
from src.echo_agent import EchoMetrics, EchoExtractionRaw, extract_echo_metrics, predict_echo_risks
from src.risk_aggregator import combine_risks
import shap
app = FastAPI(title="CVD Risk Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the React/HTML frontend
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

# ── Load Model & Preprocessor ──
device = "cpu"
preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
model_path = os.path.join(MODELS_DIR, "cvd_classifier.pt")
params_path = os.path.join(MODELS_DIR, "best_hpo_params.joblib")

model = None
preprocessor = None
feature_names = None
explainer = None
joint_model = None
joint_artifacts = None
joint_explainer = None
joint_feature_names = None
joint_kernel_explainer = None
joint_tab_dim = None

ECHO_8_DEFAULTS = {
    "LVEF": 55.0,
    "LVEDD": 50.0,
    "IVSd": 9.0,
    "LVPWd": 9.0,
    "WallMotion": 0,
    "MitralRegurgitation": 0,
    "EA_Ratio": 1.2,
    "PASP": 25.0,
    "LAVI": 28.0,
    "AorticValveArea": 3.5,
}

class _SigmoidWrapper(torch.nn.Module):
    """Wraps the classifier so SHAP sees probabilities, not logits."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        return torch.sigmoid(self.base(x))


class _JointLogitWrapper(torch.nn.Module):
    """Wraps joint-fusion model so SHAP explains raw logits (better gradients than post-sigmoid)."""
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x_tab, x_struct):
        out = self.base(x_tab, x_struct)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        if logits.ndim == 1:
            logits = logits.unsqueeze(1)
        return logits


def _joint_background_tensors(artifacts: Dict, bg_n: int = 64):
    """Build representative multi-input SHAP background from training-like data; fallback to zeros."""
    tab_features = artifacts.get("tab_features", [])
    struct_features = artifacts.get("struct_features", [])
    tab_scaler = artifacts.get("tab_scaler")
    struct_scaler = artifacts.get("struct_scaler")

    if not tab_features or not struct_features or tab_scaler is None or struct_scaler is None:
        return None, None

    raw_path = os.path.join(DATA_DIR, "raw_dataset.csv")
    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
        if all(c in df.columns for c in tab_features) and all(c in df.columns for c in struct_features):
            sample_n = min(bg_n, len(df))
            sample_df = df.sample(n=sample_n, random_state=42)
            tab_np = tab_scaler.transform(sample_df[tab_features].astype(float).values).astype(np.float32)
            struct_np = struct_scaler.transform(sample_df[struct_features].astype(float).values).astype(np.float32)
            return (
                torch.tensor(tab_np, dtype=torch.float32, device=device),
                torch.tensor(struct_np, dtype=torch.float32, device=device),
            )

    # Fallback: mean-centered baseline in scaled space
    bg_tab = torch.zeros((bg_n, len(tab_features)), dtype=torch.float32, device=device)
    bg_struct = torch.zeros((bg_n, len(struct_features)), dtype=torch.float32, device=device)
    return bg_tab, bg_struct


def _normalise_multi_input_shap(raw):
    """Normalise SHAP output for multi-input model to (tab_arr, struct_arr), each [n_samples, n_features]."""
    if isinstance(raw, list) and len(raw) == 2:
        tab_arr, struct_arr = raw[0], raw[1]
    elif isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], list) and len(raw[0]) == 2:
        tab_arr, struct_arr = raw[0][0], raw[0][1]
    else:
        raise RuntimeError(f"Unexpected multi-input SHAP output type: {type(raw)}")

    tab_arr = np.array(tab_arr)
    struct_arr = np.array(struct_arr)

    if tab_arr.ndim == 3:
        tab_arr = tab_arr[..., 0]
    if struct_arr.ndim == 3:
        struct_arr = struct_arr[..., 0]

    if tab_arr.ndim == 1:
        tab_arr = tab_arr[np.newaxis, :]
    if struct_arr.ndim == 1:
        struct_arr = struct_arr[np.newaxis, :]

    return tab_arr, struct_arr


def _joint_shap_top_factors(patient_tab_scaled: np.ndarray, patient_struct_scaled: np.ndarray, top_k: int = 3):
    """Compute top fused SHAP factors across tabular + echo inputs for one patient."""
    if joint_feature_names is None:
        return None

    tab_tensor = torch.tensor(np.atleast_2d(patient_tab_scaled).astype(np.float32), device=device)
    struct_tensor = torch.tensor(np.atleast_2d(patient_struct_scaled).astype(np.float32), device=device)

    unified_shap = None
    engine = None

    # Primary path: multi-input gradient/deep SHAP on logits
    if joint_explainer is not None:
        try:
            try:
                raw = joint_explainer.shap_values([tab_tensor, struct_tensor], check_additivity=False)
            except TypeError:
                raw = joint_explainer.shap_values([tab_tensor, struct_tensor])

            tab_shap, struct_shap = _normalise_multi_input_shap(raw)
            candidate = np.concatenate([tab_shap[0], struct_shap[0]], axis=0)

            # If gradients collapse numerically, trigger robust fallback
            if np.max(np.abs(candidate)) > 1e-8:
                unified_shap = candidate
                engine = "deep_or_gradient"
        except Exception:
            unified_shap = None

    # Fallback path: model-agnostic Kernel SHAP over concatenated (tab + struct)
    if unified_shap is None and joint_kernel_explainer is not None and joint_tab_dim is not None:
        merged = np.concatenate([
            np.atleast_2d(patient_tab_scaled).astype(np.float32),
            np.atleast_2d(patient_struct_scaled).astype(np.float32)
        ], axis=1)
        raw_kernel = joint_kernel_explainer.shap_values(merged, nsamples=120)
        if isinstance(raw_kernel, list):
            raw_kernel = raw_kernel[0]
        kernel_arr = np.array(raw_kernel)
        if kernel_arr.ndim == 1:
            kernel_arr = kernel_arr[np.newaxis, :]
        if kernel_arr.ndim == 3:
            kernel_arr = kernel_arr[..., 0]
        unified_shap = kernel_arr[0]
        engine = "kernel_fallback"

    if unified_shap is None:
        return None

    top_idx = np.argsort(np.abs(unified_shap))[-top_k:][::-1]

    return {
        "top_factors": [
            {"feature": joint_feature_names[int(i)], "impact": float(unified_shap[int(i)])}
            for i in top_idx
        ],
        "engine": engine or "unknown"
    }


def _prepare_joint_inputs(patient: "PatientData", echo_metrics: Dict):
    """Build scaled tabular and structural vectors for joint-fusion inference/SHAP."""
    if joint_artifacts is None:
        return None, None

    tab_features = joint_artifacts.get("tab_features", [])
    struct_features = joint_artifacts.get("struct_features", [])
    tab_scaler = joint_artifacts.get("tab_scaler")
    struct_scaler = joint_artifacts.get("struct_scaler")

    if not tab_features or not struct_features or tab_scaler is None or struct_scaler is None:
        return None, None

    p = patient.dict()
    tab_raw = np.array([[float(p.get(feat, 0.0)) for feat in tab_features]], dtype=np.float32)
    struct_raw = np.array([[float(echo_metrics.get(feat, 0.0)) for feat in struct_features]], dtype=np.float32)

    tab_scaled = tab_scaler.transform(tab_raw)
    struct_scaled = struct_scaler.transform(struct_raw)
    return tab_scaled.astype(np.float32), struct_scaled.astype(np.float32)

@app.on_event("startup")
def load_assets():
    global model, preprocessor, feature_names, explainer
    global joint_model, joint_artifacts, joint_explainer, joint_feature_names
    global joint_kernel_explainer, joint_tab_dim
    if os.path.exists(preprocessor_path) and os.path.exists(model_path):
        preprocessor = joblib.load(preprocessor_path)
        params = joblib.load(params_path) if os.path.exists(params_path) else {'hidden_size': 128, 'dropout': 0.3}
        
        feature_names = CONTINUOUS_FEATURES + INTERACTION_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES + PREGNANCY_FEATURES
        input_dim = len(feature_names)
        
        model = CVDClassifier(input_dim, hidden_size=params['hidden_size'], dropout=params['dropout'])
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # KernelExplainer: model-agnostic, works with BatchNorm1d layers
        # (GradientExplainer and DeepExplainer are unreliable with BatchNorm at batch_size=1)
        def _predict_prob(X_np: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                t = torch.tensor(np.atleast_2d(X_np).astype(np.float32), device=device)
                return torch.sigmoid(model(t)).cpu().numpy().flatten()

        # Use a small representative background (zeros = mean after standardisation)
        background_np = np.zeros((1, input_dim), dtype=np.float32)
        explainer = shap.KernelExplainer(_predict_prob, background_np)
        explainer._type = "kernel"
        
        print("Model, preprocessor, and SHAP explainer loaded successfully.")
    else:
        print("Warning: Model or preprocessor not found. Prediction endpoint will fail.")

    # Optional: load joint-fusion model + multi-input SHAP explainer for comprehensive predictions
    joint_model_path = os.path.join(MODELS_DIR, "cvd_joint_fusion.pt")
    joint_artifacts_path = os.path.join(MODELS_DIR, "joint_fusion_artifacts.joblib")
    if os.path.exists(joint_model_path) and os.path.exists(joint_artifacts_path):
        try:
            joint_artifacts = joblib.load(joint_artifacts_path)
            tab_features = joint_artifacts.get("tab_features", [])
            struct_features = joint_artifacts.get("struct_features", [])
            joint_feature_names = tab_features + struct_features
            joint_tab_dim = len(tab_features)

            params = joblib.load(params_path) if os.path.exists(params_path) else {'hidden_size': 128, 'dropout': 0.3}
            joint_model = JointFusionHFNet(
                tab_dim=len(tab_features),
                struct_dim=len(struct_features),
                d_model=64,
                n_heads=4,
                hidden=params.get('hidden_size', 128),
                dropout=params.get('dropout', 0.3),
            )
            joint_model.load_state_dict(torch.load(joint_model_path, map_location=device))
            joint_model.to(device)
            joint_model.eval()

            wrapped_joint = _JointLogitWrapper(joint_model).to(device)
            bg_tab, bg_struct = _joint_background_tensors(joint_artifacts, bg_n=64)
            if bg_tab is None or bg_struct is None:
                raise RuntimeError("Unable to construct joint SHAP background tensors.")

            try:
                joint_explainer = shap.DeepExplainer(wrapped_joint, [bg_tab, bg_struct])
            except Exception:
                joint_explainer = shap.GradientExplainer(wrapped_joint, [bg_tab, bg_struct])

            def _joint_predict_logits_merged(X_np: np.ndarray) -> np.ndarray:
                X_np = np.atleast_2d(X_np).astype(np.float32)
                tab_np = X_np[:, :joint_tab_dim]
                struct_np = X_np[:, joint_tab_dim:]
                with torch.no_grad():
                    tab_t = torch.tensor(tab_np, dtype=torch.float32, device=device)
                    struct_t = torch.tensor(struct_np, dtype=torch.float32, device=device)
                    out = joint_model(tab_t, struct_t)
                    logits = out[0] if isinstance(out, (tuple, list)) else out
                    if logits.ndim == 2 and logits.shape[1] == 1:
                        logits = logits.squeeze(1)
                    return logits.detach().cpu().numpy().reshape(-1)

            kernel_bg = np.concatenate([
                bg_tab.detach().cpu().numpy(),
                bg_struct.detach().cpu().numpy()
            ], axis=1)
            joint_kernel_explainer = shap.KernelExplainer(_joint_predict_logits_merged, kernel_bg)

            print("Joint-fusion model and multi-input SHAP explainer loaded successfully.")
        except Exception as e:
            joint_model = None
            joint_artifacts = None
            joint_explainer = None
            joint_feature_names = None
            joint_kernel_explainer = None
            joint_tab_dim = None
            print(f"Warning: joint-fusion assets found but failed to load: {e}")

# ── API Models ───────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    age: float
    BMI: float
    blood_pressure: float
    glucose: float
    activity: float
    cholesterol: float = 180.0       # mg/dL  (draft: Clinical — Cholesterol)
    sleep_duration: float = 7.0      # hours/night (draft: Lifestyle — Sleep duration)
    alcohol: float = 0.0             # drinks/week  (draft: Lifestyle — Alcohol consumption)
    education: int
    socioeconomic_status: int
    diet_pattern: int = 1            # 0=poor,1=moderate,2=good (draft: Lifestyle — Diet pattern)
    stress_level: int = 1            # 0=low,1=moderate,2=high  (draft: Psychosocial — Stress)
    smoking: int
    PCOS: int
    hypertension: int
    is_ever_pregnant: int
    gestational_diabetes: int = 0
    preeclampsia: int = 0
    preterm_birth: int = 0

class PredictionResponse(BaseModel):
    risk_level: str
    probability: float
    confidence_interval: str
    top_factors: List[Dict]
    disclaimer: str

# ── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/")
def read_root():
    index_path = os.path.join(_FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "CVD Risk Prediction API is running. Use /predict for inference."}

@app.get("/health")
def health_check():
    if model is None or preprocessor is None:
        return {"status": "unhealthy", "message": "Model or preprocessor missing"}
    return {"status": "healthy", "model_version": "1.0.0-elite"}


@app.get("/status/components")
def component_status():
    return {
        "status": "ok",
        "adult_model_loaded": model is not None,
        "adult_preprocessor_loaded": preprocessor is not None,
        "adult_shap_loaded": explainer is not None,
        "joint_model_loaded": joint_model is not None,
        "joint_artifacts_loaded": joint_artifacts is not None,
        "joint_shap_loaded": joint_explainer is not None,
        "joint_kernel_shap_loaded": joint_kernel_explainer is not None,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PatientData):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    input_df = pd.DataFrame([data.dict()])
    from src.preprocess import add_interaction_features, apply_pregnancy_gate
    input_df = apply_pregnancy_gate(input_df)
    input_df = add_interaction_features(input_df)
    
    # Ensure all columns present
    cols = CONTINUOUS_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES + PREGNANCY_FEATURES
    for c in cols:
        if c not in input_df.columns:
            input_df[c] = 0
            
    X_transformed = preprocessor.transform(input_df)
    X_tensor = torch.FloatTensor(X_transformed)
    
    # Use temperature = 2.0 to safely scale extremes into valid percentages without changing classes
    mean_prob, std_uncert = mc_dropout_predict(model, X_tensor, n_passes=50, device=device, temperature=2.0)
    prob = float(mean_prob[0])
    std  = float(std_uncert[0])
    risk_lvl = prob_to_label(prob)
    ci_str   = get_risk_with_confidence(prob, std)
    
    # Real SHAP Explainability
    try:
        etype = getattr(explainer, "_type", "gradient")
        if etype == "kernel":
            raw = explainer.shap_values(X_tensor.cpu().numpy(), nsamples=50)
        else:
            # GradientExplainer / DeepExplainer — pass tensor
            raw = explainer.shap_values(X_tensor)
        # Normalise shape → (n_samples, n_features)
        if isinstance(raw, list):
            raw = raw[0]
        shap_arr = np.array(raw)
        while shap_arr.ndim > 2:
            shap_arr = shap_arr[..., 0]
        if shap_arr.ndim == 1:
            shap_arr = shap_arr[np.newaxis, :]
        row = shap_arr[0]
        top_idx = np.argsort(np.abs(row))[-3:][::-1]
        top_factors = [
            {"feature": feature_names[int(i)], "impact": float(row[i])}
            for i in top_idx
        ]
    except Exception as e:
        top_factors = [{"feature": "SHAP unavailable", "impact": 0.0, "error": str(e)}]
    
    return {
        "risk_level": risk_lvl,
        "probability": round(prob, 4),
        "confidence_interval": ci_str,
        "top_factors": top_factors,
        "disclaimer": "This is a research screening tool. Consult a licensed physician for diagnosis."
    }
@app.post("/predict_comprehensive")
async def predict_comprehensive(patient_data: str = Form(...), echo_report: UploadFile = File(...)):
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    data_dict = json.loads(patient_data)
    try:
        patient = PatientData(**data_dict)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    base_pred = predict(patient)
    lifestyle_prob = base_pred["probability"]
    
    pdf_bytes = await echo_report.read()

    # Extraction pipeline: text→regex (primary) + LLM (enhancement) with per-field reconciliation.
    # Falls back to per-field clinical defaults ONLY for fields genuinely absent from the PDF.
    echo_metrics, extraction_status, extraction_warning, echo_fields_extracted, echo_fields_defaulted = extract_echo_metrics(pdf_bytes)
        
    model_path = os.path.join(MODELS_DIR, "echo_xgboost.pkl")
    echo_probs = predict_echo_risks(echo_metrics, model_path=model_path)
    
    echo_metrics_dict = {**ECHO_8_DEFAULTS, **echo_metrics.model_dump()}
    # Raw extracted view for UI transparency: fields that were defaulted are shown as null.
    echo_metrics_extracted_raw = dict(echo_metrics_dict)
    for field in echo_fields_defaulted:
        if field in echo_metrics_extracted_raw:
            echo_metrics_extracted_raw[field] = None
    final_output = combine_risks(lifestyle_prob, echo_probs, echo_metrics_dict)

    # Multi-input SHAP for fused interpretability (tabular + echo)
    top_factors = base_pred["top_factors"]
    shap_engine = "tabular_kernel_fallback"
    try:
        tab_scaled, struct_scaled = _prepare_joint_inputs(patient, echo_metrics_dict)
        if tab_scaled is not None and struct_scaled is not None and joint_explainer is not None:
            fused_top = _joint_shap_top_factors(tab_scaled, struct_scaled, top_k=3)
            if fused_top and isinstance(fused_top, dict):
                top_factors = fused_top.get("top_factors", top_factors)
                shap_engine = fused_top.get("engine", shap_engine)
    except Exception as e:
        # Preserve API availability if SHAP fails in runtime
        shap_engine = "error"
        top_factors = base_pred["top_factors"] + [{"feature": "joint_shap_warning", "impact": 0.0, "error": str(e)}]
    
    return {
        "risk_level": final_output["risk_level"],
        "probability": final_output["final_probability"],
        "confidence_interval": base_pred["confidence_interval"],
        "top_factors": top_factors,
        "shap_engine": shap_engine,
        "disclaimer": final_output["recommendation"],
        "lifestyle_risk": final_output.get("lifestyle_risk"),
        "echo_model_max_risk": final_output.get("echo_model_max_risk"),
        "echo_structural_risk": final_output.get("echo_structural_risk"),
        "echo_metrics": echo_metrics_dict,
        "echo_metrics_extracted_raw": echo_metrics_extracted_raw,
        "echo_metrics_used": list(ECHO_8_DEFAULTS.keys()),
        "echo_metrics_used_count": len(ECHO_8_DEFAULTS),
        "echo_extraction_status": extraction_status,
        "echo_extraction_warning": extraction_warning,
        "echo_fields_extracted": echo_fields_extracted,
        "echo_fields_defaulted": echo_fields_defaulted,
        "echo_disease_risks": final_output["echo_risks"]
    }

@app.post("/explain")
def explain(data: PatientData):
    # dedicated SHAP endpoint
    pred_response = predict(data)
    return {
        "risk_level": pred_response["risk_level"],
        "probability": pred_response["probability"],
        "top_shap_factors": pred_response["top_factors"]
    }
