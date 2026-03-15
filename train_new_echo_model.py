import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

np.random.seed(42)
n_samples = 5000

# Features
lvef = np.random.normal(loc=55, scale=12, size=n_samples).clip(15, 75)
lvedd = np.random.normal(loc=50, scale=8, size=n_samples).clip(35, 80)
wall_motion = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1], size=n_samples)
mitral_regurg = np.random.choice([0, 1, 2, 3], p=[0.5, 0.3, 0.15, 0.05], size=n_samples)

# NEW FEATURES
# E/A Ratio: Normal 1.0-2.0. <1.0 or >2.0 indicates diastolic dysfunction (Heart Failure)
ea_ratio = np.random.normal(loc=1.2, scale=0.5, size=n_samples).clip(0.3, 3.5)
# PASP (Pulmonary Artery Systolic Pressure): Normal 15-30. >35 is pulmonary hypertension
pasp = np.random.normal(loc=25, scale=10, size=n_samples).clip(10, 80)
# LAVI (Left Atrial Volume Index): Normal 16-34. >34 is enlarged.
lavi = np.random.normal(loc=28, scale=12, size=n_samples).clip(10, 80)
# Aortic Valve Area (cm2): Normal 3.0-4.0. <1.0 is severe stenosis.
aortic_valve_area = np.random.normal(loc=3.5, scale=1.0, size=n_samples).clip(0.5, 5.0)

X = pd.DataFrame({
    'LVEF': lvef,
    'LVEDD': lvedd,
    'WallMotion': wall_motion,
    'MitralRegurgitation': mitral_regurg,
    'EA_Ratio': ea_ratio,
    'PASP': pasp,
    'LAVI': lavi,
    'AorticValveArea': aortic_valve_area
})

# Targets (Probabilities based on heuristic rules)
# Heart Failure incorporates Diastolic Dysfunction (EA Ratio), Pulmonary Hypertension (PASP), and Atrial enlargement (LAVI)
hf_risk_factors = (-5 + 0.1*(60 - lvef) + 0.1*(lvedd - 50) + 0.5*mitral_regurg
                   + 0.5*(ea_ratio < 0.8) + 0.5*(ea_ratio > 2.0)
                   + 0.05*(pasp - 30) + 0.05*(lavi - 34))

prob_hf = 1 / (1 + np.exp(-hf_risk_factors))

# CAD heavily relies on Wall Motion, but we'll add slight influence from strict Aortic Stenosis blocking flow
cad_risk_factors = -4 + 1.5*wall_motion + 0.02*(60 - lvef) + 0.5*(aortic_valve_area < 1.0)
prob_cad = 1 / (1 + np.exp(-cad_risk_factors))

# Cardiomyopathy
cmp_risk_factors = -6 + 0.15*(55 - lvef) + 0.2*(lvedd - 55) + 0.04*(lavi - 34)
prob_cmp = 1 / (1 + np.exp(-cmp_risk_factors))

Y = pd.DataFrame({
    'HeartFailure': (np.random.rand(n_samples) < prob_hf).astype(int),
    'CAD': (np.random.rand(n_samples) < prob_cad).astype(int),
    'Cardiomyopathy': (np.random.rand(n_samples) < prob_cmp).astype(int)
})

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

models = {}
for col in Y.columns:
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, Y_train[col])
    models[col] = model

export_package = {
    'features': ['LVEF', 'LVEDD', 'WallMotion', 'MitralRegurgitation', 'EA_Ratio', 'PASP', 'LAVI', 'AorticValveArea'],
    'models': models
}

filename = 'models/echo_xgboost.pkl'
with open(filename, 'wb') as f:
    pickle.dump(export_package, f)

print(f"Saved enhanced 8-feature ML model to {filename}")
