import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.risk_aggregator import combine_risks


def test_severe_echo_not_low_risk():
    result = combine_risks(
        lifestyle_prob=0.09,
        echo_probs={"HeartFailure": 0.08, "CAD": 0.05, "Cardiomyopathy": 0.07},
        echo_metrics={"LVEF": 42.0, "LVEDD": 58.0, "WallMotion": 1, "MitralRegurgitation": 2},
    )

    assert result["risk_level"] in {"Moderate", "High"}
    assert result["final_probability"] >= 0.58


def test_critical_echo_escalates_to_high_floor():
    result = combine_risks(
        lifestyle_prob=0.05,
        echo_probs={"HeartFailure": 0.10, "CAD": 0.08, "Cardiomyopathy": 0.07},
        echo_metrics={"LVEF": 35.0, "LVEDD": 62.0, "WallMotion": 2, "MitralRegurgitation": 3},
    )

    assert result["risk_level"] == "High"
    assert result["final_probability"] >= 0.72


def test_probability_bounds_with_extreme_inputs():
    result = combine_risks(
        lifestyle_prob=2.5,
        echo_probs={"HeartFailure": 4.0, "CAD": -1.0, "Cardiomyopathy": 0.3},
        echo_metrics={"LVEF": 55.0, "LVEDD": 48.0, "WallMotion": 0, "MitralRegurgitation": 0},
    )

    assert 0.0 <= result["final_probability"] <= 1.0
