# Ethical AI Statement — CVD Risk Predictor
**Project Title**: Cardiovascular Risk Analysis for Young Women (18–35)

## 1. Bias & Fairness
Healthcare datasets often suffer from historical biases. We have implemented several mitigations:
- **Fairness Audit**: Used Fairlearn to audit predictions across socioeconomic status (SES) and age buckets. The goal is to minimize the "Equal Opportunity Difference" to <0.05.
- **Stratified Training**: All models are trained using `StratifiedKFold` to ensure class distribution (Low/Moderate/High risk) is preserved across folds.
- **Population Representation**: Used NHANES as a reference distribution for the synthetic generator to avoid over-sampling high-income groups.

## 2. Privacy & Data Security
- **De-identification**: No Personally Identifiable Information (PII) is stored. All training data consists of de-identified clinical and demographic records.
- **Synthetic Data**: The utilize of CTGAN allows for data augmentation and analysis without exposing raw patient records, reducing the risk of privacy leaks from the training set.

## 3. Transparency & Explainability
- **Explainable AI (XAI)**: Every prediction is accompanied by a SHAP waterfall plot and a list of top contributing factors.
- **Uncertainty Quantification**: Using MC Dropout, we report a 95% Confidence Interval for every risk probability. Doctors are encouraged to view lower-confidence results (e.g., ±0.15) with skepticism.

## 4. Clinical Responsibility Disclaimer
- **Not a Diagnostic Tool**: This software is intended for **research and screening purposes only**. It does not replace a clinical examination by a licensed medical professional.
- **Human-in-the-Loop**: The system is designed to provide *decision support* to physicians, not to act as an autonomous decision-maker.

## 5. Intended Use
This tool is specifically calibrated for **women aged 18–35**. Its use on other demographics (e.g., men or older populations) is likely to result in inaccurate and misleading predictions.
