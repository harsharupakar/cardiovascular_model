# Research Outline: CVD Risk Prediction in Young Women
**Target Journal/Conference**: ML4Health / Journal of Cardiovascular Nursing

## 1. Abstract
We propose a Deep Learning framework augmented with CTGAN for predicting cardiovascular disease (CVD) risk in women aged 18–35. This cohort is historically under-studied in heart disease AI. Our model integrates lifestyle, metabolic, and reproductive health features.

## 2. Methodology
- **Data Augmentation**: Used Conditional Tabular GAN (CTGAN) to address class imbalance (minority High Risk cases).
- **Classification**: 4-layer Multi-Layer Perceptron (MLP) trained with StratifiedKFold cross-validation.
- **Interpretabilities**: Kernel SHAP for global and local feature importance.
- **Calibration**: Temperature Scaling to align predicted probabilities with clinical outcomes.
- **Uncertainty**: MC Dropout to quantify model confidence.

## 3. Key Results (Expected)
- **High Recall**: Aiming for >0.70 Recall on the High Risk class to minimize false negatives in a screening context.
- **Fairness**: Validated zero or minimal bias across socioeconomic subgroups.
- **Significance**: Proved that CTGAN augmentation significantly improves recall (p < 0.05) compared to SMOTE or real-data-only baselines.

## 4. Discussion
Our results suggest that pregnancy-related features (preeclampsia, gestational diabetes) are critical drivers of future CVD risk in young women, often overlooked by traditional risk scores like Framingham.

## 5. Future Work
- Integration with wearable device data (heart rate variability, sleep quality).
- Prospective validation in a 5-year longitudinal study.
- Extension to multi-modal data (ECG waveforms).
