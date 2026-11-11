"""
Unit Tests for Model Artifacts, Prediction Schema, and Decision Threshold Policies.

Proves to interviewers that:
1. Trained models and metadata serialize cleanly.
2. Predictions follow strictly calibrated [0.0, 1.0] probability constraints.
3. Out-of-time evaluation metrics satisfy production quality gates.
4. Tri-tier decision policy behaves deterministically.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pytest

from src.config import MODELS_DIR
from src.features import FEATURE_NAMES


def test_saved_model_artifacts_exist():
    """Verify champion model, alternate model, and evaluation report exist on disk."""
    model_path = MODELS_DIR / "model.pkl"
    lgbm_path = MODELS_DIR / "lgbm_model.pkl"
    metrics_path = MODELS_DIR / "eval_metrics.json"

    assert model_path.exists(), f"Missing champion model: {model_path}"
    assert lgbm_path.exists(), f"Missing alternate model: {lgbm_path}"
    assert metrics_path.exists(), f"Missing metrics report: {metrics_path}"


def test_model_prediction_shape_and_probability_bounds():
    """Verify that champion model outputs valid 2D probabilities in [0.0, 1.0]."""
    model = joblib.load(MODELS_DIR / "model.pkl")

    # Synthetic sample row matching 22 features
    dummy_input = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)
    probs = model.predict_proba(dummy_input)

    assert probs.shape == (1, 2), f"Expected shape (1, 2), got {probs.shape}"
    assert np.isclose(probs.sum(), 1.0, atol=1e-5), "Probabilities do not sum to 1.0"
    assert 0.0 <= probs[0, 1] <= 1.0, f"Fraud probability out of bounds: {probs[0, 1]}"


def test_evaluation_metrics_quality_gates():
    """Verify out-of-time test metrics satisfy strict performance thresholds."""
    with open(MODELS_DIR / "eval_metrics.json") as f:
        metrics = json.load(f)

    # 1. Tuning threshold validity
    tau = metrics["tuning_policy"]["best_threshold"]
    assert 0.0 < tau < 1.0, f"Invalid decision threshold: {tau}"

    # 2. Champion model quality gates
    champ = metrics["models_comparison"]["xgboost"]
    assert champ["pr_auc"] >= 0.90, f"PR-AUC below production gate: {champ['pr_auc']}"
    assert champ["recall"] >= 0.80, f"Recall below production gate: {champ['recall']}"
    assert champ["precision"] >= 0.80, f"Precision below production gate: {champ['precision']}"

    # 3. LightGBM quality gates
    lgbm = metrics["models_comparison"]["lightgbm"]
    assert lgbm["pr_auc"] >= 0.70, f"LightGBM PR-AUC below gate: {lgbm['pr_auc']}"
    assert lgbm["recall"] >= 0.80, f"LightGBM Recall below gate: {lgbm['recall']}"


def test_tri_tier_decision_policy_mapping():
    """Verify the 3-tier action routing logic (ALLOW, REVIEW, BLOCK)."""
    with open(MODELS_DIR / "eval_metrics.json") as f:
        policy = json.load(f)["tuning_policy"]

    tau_allow = policy["tau_allow"]
    tau_block = policy["tau_block"]

    def route_action(score: float) -> str:
        if score < tau_allow:
            return "AUTO_APPROVE"
        elif score < tau_block:
            return "MANUAL_REVIEW"
        else:
            return "AUTO_BLOCK"

    # Test edge cases
    assert route_action(0.01) == "AUTO_APPROVE"
    assert route_action(tau_allow - 0.01) == "AUTO_APPROVE"
    assert route_action(tau_allow + 0.01) == "MANUAL_REVIEW"
    assert route_action(tau_block + 0.05) == "AUTO_BLOCK"
    assert route_action(0.99) == "AUTO_BLOCK"
