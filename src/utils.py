"""
utils.py — Shared helpers: seeds, environment logging, constants.
"""
import os
import random
import sys
import numpy as np
import torch

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42

def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def log_environment():
    import platform
    print("=" * 50)
    print(f"Python  : {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Torch   : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda if torch.cuda.is_available() else 'CPU only'}")
    try:
        import sdv; print(f"SDV     : {sdv.__version__}")
    except ImportError:
        pass
    try:
        import optuna; print(f"Optuna  : {optuna.__version__}")
    except ImportError:
        pass
    print("=" * 50)

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT_DIR, "data")
MODELS_DIR  = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")

for _d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
    os.makedirs(_d, exist_ok=True)

# ─── Feature Groups ──────────────────────────────────────────────────────────
CONTINUOUS_FEATURES = [
    "age", "BMI", "blood_pressure", "glucose", "activity",
    "cholesterol", "sleep_duration", "alcohol"
]
ORDINAL_FEATURES = ["education", "socioeconomic_status", "diet_pattern", "stress_level"]
BINARY_FEATURES  = ["smoking", "PCOS", "hypertension", "is_ever_pregnant"]
PREGNANCY_FEATURES = ["gestational_diabetes", "preeclampsia", "preterm_birth"]

ALL_BASE_FEATURES = CONTINUOUS_FEATURES + ORDINAL_FEATURES + BINARY_FEATURES + PREGNANCY_FEATURES

INTERACTION_FEATURES = [
    "BMI_x_BP", "Glucose_x_BMI", "PCOS_x_BMI", "Age_x_PhysicalActivity"
]

TARGET_COL = "cvd_risk"

# ─── Risk thresholds ─────────────────────────────────────────────────────────
LOW_THRESHOLD  = 0.33
HIGH_THRESHOLD = 0.66

def prob_to_label(prob: float) -> str:
    if prob < LOW_THRESHOLD:
        return "Low"
    elif prob < HIGH_THRESHOLD:
        return "Moderate"
    return "High"
