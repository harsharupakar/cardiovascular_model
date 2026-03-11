# Prospective Clinical Validation Study Design

To transition from a research screening tool to a clinically validated medical device, the following prospective study is proposed.

## 1. Study Objectives
- **Primary**: To evaluate the accuracy of the GAN-Augmented MLP in predicting incident hypertension or CAD within 5 years of baseline.
- **Secondary**: To compare model performance against the Framingham Risk Score in the 18–35 female cohort.

## 2. Study Population
- **Size**: 1,000 women aged 18–35.
- **Inclusion**: Healthy at baseline (no pre-existing CAD or Stroke).
- **Diversity**: Multi-center recruitment ensuring representative SES and ethnic backgrounds.

## 3. Methodology
1. **Baseline**: Perform 16-feature screening (clinical, lifestyle, reproductive history).
2. **Prediction**: Run the Model and record Risk Score + Confidence Interval.
3. **Follow-up**: Annual check-ups for 5 years to track incident CVD events.
4. **Endpoint Analysis**: Compare baseline AI risk buckets (Low/Mod/High) against actual 5-year event rates.

## 4. Institutional Review Board (IRB) & Ethics
- **Consent**: Full informed consent required regarding the experimental nature of the AI algorithm.
- **Transparency**: Patients provided with their SHAP explanations to ensure they understand the drivers of their risk score.
- **Safety**: High-risk labels must be immediately shared with the participant's primary care physician for clinical follow-up.

## 5. Success Metrics
- **AUROC > 0.80** for 5-year event prediction.
- **Equal Opportunity Difference < 0.05** across recruitment sites.
