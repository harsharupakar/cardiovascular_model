# Limitations of the CVD Risk Model

As with any machine learning research in healthcare, our system has several limitations that must be addressed in future work.

## 1. Data Source & Generalizability
- **NHANES Calibration**: The training data is calibrated to US NHANES distributions. The model's performance on international populations (e.g., in South Asia or Sub-Saharan Africa) may be significantly different due to varying genetic and environmental risk factors.
- **Label Proxy**: Since incident CVD is rare in the 18–35 age group, we use a composite proxy (hypertension, metabolic syndrome markers, and reproductive complications) as a surrogate for "High Risk." This proxy may not perfectly correlate with long-term cardiovascular events.

## 2. Methodology Caveats
- **Synthetic Data (CTGAN)**: While CTGAN improves recall, synthetic records are statistical approximations. They may miss extremely rare edge cases or complex feature interactions not captured during the generator's training.
- **Cross-Sectional vs. Longitudinal**: Our model is trained on cross-sectional data. A strictly "risk" model should ideally be trained on longitudinal cohort data to predict future events.

## 3. Implementation Constraints
- **SHAP Computation**: Kernel SHAP is computationally expensive, currently limiting its real-time use in the FastAPI backend for complex patients.
- **External Validation Gap**: The Kaggle CVD dataset used for external validation lacks many of our engineered features (e.g., pregnancy details), requiring us to validate only on a subset of the model's capabilities.

## 4. Ethical Considerations
- **Demographic Parity**: While we audit for fairness, the model may still exhibit unintended performance gaps in specific intersections (e.g., low SES + specific age group) due to small sample sizes in those strata.
