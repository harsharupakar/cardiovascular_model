def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _echo_structural_score(echo_metrics: dict | None) -> float:
    if not echo_metrics:
        return 0.0

    lvef = float(echo_metrics.get("LVEF", 55.0))
    lvedd = float(echo_metrics.get("LVEDD", 50.0))
    ivsd = float(echo_metrics.get("IVSd", 9.0))
    lvpwd = float(echo_metrics.get("LVPWd", 9.0))
    wall_motion = float(echo_metrics.get("WallMotion", 0.0))
    mitral_regurg = float(echo_metrics.get("MitralRegurgitation", 0.0))
    ea_ratio = float(echo_metrics.get("EA_Ratio", 1.2))
    pasp = float(echo_metrics.get("PASP", 25.0))
    lavi = float(echo_metrics.get("LAVI", 28.0))
    aortic_valve_area = float(echo_metrics.get("AorticValveArea", 3.5))

    if lvef <= 35:
        lvef_score = 1.0
    elif lvef <= 45:
        lvef_score = 0.82
    elif lvef <= 50:
        lvef_score = 0.58
    else:
        lvef_score = 0.12

    if lvedd >= 60:
        lvedd_score = 1.0
    elif lvedd >= 55:
        lvedd_score = 0.78
    elif lvedd >= 50:
        lvedd_score = 0.45
    else:
        lvedd_score = 0.10

    wall_score = _clamp_01(wall_motion / 2.0)
    mr_score = _clamp_01(mitral_regurg / 3.0)

    # Wall thickness / hypertrophy signal (young female relevance: HCM / chronic HTN / athletic remodeling)
    septal_score = 1.0 if ivsd >= 13.0 else (0.55 if ivsd >= 11.0 else 0.0)
    posterior_score = 1.0 if lvpwd >= 13.0 else (0.55 if lvpwd >= 11.0 else 0.0)
    wall_thickness_score = max(septal_score, posterior_score)

    # Diastolic & Pulm
    ea_score = 1.0 if ea_ratio < 0.8 or ea_ratio > 2.0 else 0.0
    pasp_score = _clamp_01((pasp - 30) / 40.0)
    lavi_score = _clamp_01((lavi - 34) / 46.0)
    
    # Aortic Valve
    ava_score = 1.0 if aortic_valve_area < 1.0 else (0.5 if aortic_valve_area < 1.5 else 0.0)

    weighted = (
        0.27 * lvef_score +
        0.13 * lvedd_score +
        0.13 * wall_score +
        0.10 * mr_score +
        0.10 * ea_score +
        0.05 * pasp_score +
        0.05 * lavi_score +
        0.09 * ava_score +
        0.08 * wall_thickness_score
    )
    return _clamp_01(weighted)


def combine_risks(lifestyle_prob: float, echo_probs: dict, echo_metrics: dict | None = None) -> dict:
    """
    Fuses lifestyle/tabular risk with echo-derived risk.
    Echo findings are given dominant weight with severity guardrails.
    """
    lifestyle_prob = _clamp_01(lifestyle_prob)
    max_echo_model_prob = max(echo_probs.values()) if echo_probs else 0.0
    max_echo_model_prob = _clamp_01(max_echo_model_prob)

    structural_score = _echo_structural_score(echo_metrics)

    # Robust echo score: use stronger signal between model and direct structural severity.
    echo_signal = max(max_echo_model_prob, structural_score)

    # Echo dominates final score in comprehensive mode.
    final_prob = 0.25 * lifestyle_prob + 0.75 * echo_signal

    if echo_metrics:
        lvef = float(echo_metrics.get("LVEF", 55.0))
        lvedd = float(echo_metrics.get("LVEDD", 50.0))
        ivsd = float(echo_metrics.get("IVSd", 9.0))
        lvpwd = float(echo_metrics.get("LVPWd", 9.0))
        wall_motion = float(echo_metrics.get("WallMotion", 0.0))
        mitral_regurg = float(echo_metrics.get("MitralRegurgitation", 0.0))
        ea_ratio = float(echo_metrics.get("EA_Ratio", 1.2))
        pasp = float(echo_metrics.get("PASP", 25.0))
        lavi = float(echo_metrics.get("LAVI", 28.0))
        aortic_valve_area = float(echo_metrics.get("AorticValveArea", 3.5))

        severe_combo = lvef <= 45 and (lvedd >= 55 or mitral_regurg >= 2 or wall_motion >= 1 or pasp >= 40 or ivsd >= 11.0 or lvpwd >= 11.0)
        critical_flag = lvef <= 40 or wall_motion >= 2 or mitral_regurg >= 3 or aortic_valve_area <= 1.0 or ea_ratio > 2.5 or ivsd >= 13.0 or lvpwd >= 13.0

        if severe_combo:
            final_prob = max(final_prob, 0.58)
        if critical_flag:
            final_prob = max(final_prob, 0.72)

    final_prob = _clamp_01(final_prob)

    if final_prob >= 0.7:
        risk_level = "High"
        recommendation = "Urgent Cardiologist consultation advised. Structural abnormalities detected."
    elif final_prob >= 0.4:
        risk_level = "Moderate"
        recommendation = "Cardiologist consultation advised. Elevated combined risk profile."
    else:
        risk_level = "Low"
        recommendation = "Routine follow-up. Low combined risk."

    return {
        "final_probability": round(final_prob, 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "lifestyle_risk": round(lifestyle_prob, 4),
        "echo_model_max_risk": round(max_echo_model_prob, 4),
        "echo_structural_risk": round(structural_score, 4),
        "echo_risks": {k: round(v, 4) for k, v in echo_probs.items()}
    }
