import json
import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app.main as main_mod


client = TestClient(main_mod.app)


class _DummyEchoMetrics:
    def model_dump(self):
        return {
            "LVEF": 42.0,
            "LVEDD": 58.0,
            "WallMotion": 1,
            "MitralRegurgitation": 2,
        }


def test_predict_comprehensive_contract(monkeypatch):
    monkeypatch.setattr(main_mod, "model", object())
    monkeypatch.setattr(main_mod, "preprocessor", object())

    monkeypatch.setattr(
        main_mod,
        "predict",
        lambda _patient: {
            "risk_level": "Low",
            "probability": 0.11,
            "confidence_interval": "0.11 ± 0.06",
            "top_factors": [{"feature": "BMI", "impact": 0.02}],
            "disclaimer": "base",
        },
    )
    monkeypatch.setattr(main_mod, "extract_text_from_pdf_bytes", lambda _b: "echo report text")
    monkeypatch.setattr(main_mod, "parse_metrics_with_gemini", lambda _t: _DummyEchoMetrics())
    monkeypatch.setattr(
        main_mod,
        "predict_echo_risks",
        lambda _m, model_path=None: {"HeartFailure": 0.08, "CAD": 0.12, "Cardiomyopathy": 0.07},
    )
    monkeypatch.setattr(
        main_mod,
        "combine_risks",
        lambda lifestyle_prob, echo_probs, echo_metrics: {
            "risk_level": "Moderate",
            "final_probability": 0.58,
            "recommendation": "Cardiologist consultation advised.",
            "lifestyle_risk": lifestyle_prob,
            "echo_model_max_risk": max(echo_probs.values()),
            "echo_structural_risk": 0.725,
            "echo_risks": echo_probs,
        },
    )

    payload = {
        "age": 30,
        "BMI": 28,
        "blood_pressure": 122,
        "glucose": 98,
        "activity": 3,
        "cholesterol": 185,
        "sleep_duration": 7,
        "alcohol": 0,
        "education": 2,
        "socioeconomic_status": 2,
        "diet_pattern": 1,
        "stress_level": 1,
        "smoking": 0,
        "PCOS": 0,
        "hypertension": 0,
        "is_ever_pregnant": 1,
        "gestational_diabetes": 0,
        "preeclampsia": 0,
        "preterm_birth": 0,
    }

    response = client.post(
        "/predict_comprehensive",
        data={"patient_data": json.dumps(payload)},
        files={"echo_report": ("mock_echo_report.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] == "Moderate"
    assert data["probability"] == 0.58
    assert "top_factors" in data and isinstance(data["top_factors"], list)
    assert data["echo_metrics"]["LVEF"] == 42.0
    assert data["lifestyle_risk"] == 0.11
    assert data["echo_model_max_risk"] == 0.12
    assert data["echo_structural_risk"] == 0.725
    assert "shap_engine" in data


def test_predict_comprehensive_uses_joint_echo_top_factors(monkeypatch):
    monkeypatch.setattr(main_mod, "model", object())
    monkeypatch.setattr(main_mod, "preprocessor", object())

    monkeypatch.setattr(
        main_mod,
        "predict",
        lambda _patient: {
            "risk_level": "Low",
            "probability": 0.10,
            "confidence_interval": "0.10 ± 0.05",
            "top_factors": [{"feature": "BMI", "impact": 0.02}],
            "disclaimer": "base",
        },
    )
    monkeypatch.setattr(main_mod, "extract_text_from_pdf_bytes", lambda _b: "echo report text")
    monkeypatch.setattr(main_mod, "parse_metrics_with_gemini", lambda _t: _DummyEchoMetrics())
    monkeypatch.setattr(
        main_mod,
        "predict_echo_risks",
        lambda _m, model_path=None: {"HeartFailure": 0.08, "CAD": 0.12, "Cardiomyopathy": 0.07},
    )
    monkeypatch.setattr(
        main_mod,
        "combine_risks",
        lambda lifestyle_prob, echo_probs, echo_metrics: {
            "risk_level": "Moderate",
            "final_probability": 0.58,
            "recommendation": "Cardiologist consultation advised.",
            "lifestyle_risk": lifestyle_prob,
            "echo_model_max_risk": max(echo_probs.values()),
            "echo_structural_risk": 0.725,
            "echo_risks": echo_probs,
        },
    )

    monkeypatch.setattr(
        main_mod,
        "_prepare_joint_inputs",
        lambda patient, echo_metrics: (
            [[0.0] * 19],
            [[0.0] * 4],
        ),
    )
    monkeypatch.setattr(
        main_mod,
        "joint_explainer",
        object(),
    )
    monkeypatch.setattr(
        main_mod,
        "_joint_shap_top_factors",
        lambda _tab, _struct, top_k=3: {
            "top_factors": [
                {"feature": "LVEDD", "impact": 1.20},
                {"feature": "LVEF", "impact": 1.10},
                {"feature": "MitralRegurgitation", "impact": 0.53},
            ],
            "engine": "deep_or_gradient",
        },
    )

    payload = {
        "age": 30,
        "BMI": 28,
        "blood_pressure": 122,
        "glucose": 98,
        "activity": 3,
        "cholesterol": 185,
        "sleep_duration": 7,
        "alcohol": 0,
        "education": 2,
        "socioeconomic_status": 2,
        "diet_pattern": 1,
        "stress_level": 1,
        "smoking": 0,
        "PCOS": 0,
        "hypertension": 0,
        "is_ever_pregnant": 1,
        "gestational_diabetes": 0,
        "preeclampsia": 0,
        "preterm_birth": 0,
    }

    response = client.post(
        "/predict_comprehensive",
        data={"patient_data": json.dumps(payload)},
        files={"echo_report": ("mock_echo_report.pdf", b"%PDF-1.4 test", "application/pdf")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["top_factors"][0]["feature"] == "LVEDD"
    assert data["shap_engine"] == "deep_or_gradient"
    assert any(f["feature"] in {"LVEF", "LVEDD", "MitralRegurgitation"} for f in data["top_factors"])
