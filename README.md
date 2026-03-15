# Cardiovascular Risk Prediction Model

This repository contains an elite, Ph.D.-level pipeline for cardiovascular disease risk prediction targeting women aged 18-35.

## Data Preprocessing
- Applies strict pregnancy gating: reproductive complications are forced to `0` if `is_ever_pregnant = 0`.
- Standardized Continuous Features: `age`, `BMI`, `blood_pressure`, `glucose`, `activity`.
- Standardized Binary Features: `smoking`, `PCOS`, `hypertension`, `is_ever_pregnant`.
- Engineered Interaction Features: `BMI_x_BP`, `Glucose_x_BMI`, `PCOS_x_BMI`, `Age_x_PhysicalActivity`.
- Artifacts (scalers, encoders) saved to `models/preprocessor.pkl`.

## Synthetic Data Generation
- Utilizes `CTGANSynthesizer` (with `TVAESynthesizer` fallback) focused on the High-Risk minority class to prevent class imbalance.
- Fidelity Gating: Models that produce poor similarities (KS/chi-squared tests via `sdv`) falling below thresholds are discarded. 
- Generated artifacts and reports are saved to `outputs/fidelity/`.

## Classifier Architecture
- PyTorch Multi-Layer Perceptron (MLP) trained with `BCEWithLogitsLoss` and class-imbalanced `pos_weight`.
- Validation splits and early stopping (patience = 10 epochs).
- Final model saved to `models/cvd_classifier.pt`.

## Evaluation & Fairness
- **Metrics**: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Explainability**: Real `SHAP` values via `DeepExplainer` (with `KernelExplainer` fallback) providing Global Beeswarm and Local Waterfall visualizations to `outputs/shap/`.
- **Fairness**: Automated audits using `fairlearn` to assess Equal Opportunity and Demographic Parity differences across Socioeconomic Status and explicit Age Buckets (18-22, 23-27, 28-35). Exported to `outputs/fairness/`.

## API Endpoints
Provided via `FastAPI` in `app/main.py`:
- `GET /health` : Health check of the service.
- `POST /predict` : Returns the predicted risk level, probability score, and top 3 personalized `SHAP` factors for a given patient vector.
- `POST /explain` : Dedicated explanation-only endpoint mapping back to the prediction data scheme.

## Testing Setup
Run rigorous pipeline assertions locally utilizing `pytest`:
```bash
python -m pytest tests/
```
Tests ensure that the pregnancy gating remains logically sound, CTGAN fallback functions correctly on minority classes, PyTorch gradients flow through logits correctly without early sigmoid activations, and FastAPI pydantic schemas hold correctly.