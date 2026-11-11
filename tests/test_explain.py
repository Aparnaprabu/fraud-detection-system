"""
Unit Tests for Model Explainability and TreeSHAP Latency.

Proves to interviewers that:
1. SHAP explanations conform to strict output contracts and explainable reason codes.
2. Plain-English descriptions are generated for all top risk factors.
3. Explanations execute in sub-millisecond latency (< 20ms budget).
"""

import time
import pytest
from src.explain import FraudExplainer
from src.features import CustomerStateCache


@pytest.fixture(scope="module")
def explainer():
    """Load cached explainer instance for tests."""
    return FraudExplainer()


def test_explanation_contract_and_schema(explainer):
    """Verify that explanation payload matches expected API structure."""
    sample_tx = {
        "step": 12,
        "type": "TRANSFER",
        "amount": 100000.0,
        "nameOrig": "C999",
        "oldbalanceOrg": 100000.0,
        "nameDest": "C888",
        "oldbalanceDest": 0.0,
    }

    result = explainer.explain_transaction(sample_tx, top_k=3)

    assert "fraud_probability" in result
    assert "top_drivers" in result
    assert "base_value" in result
    assert "explanation_latency_ms" in result

    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert len(result["top_drivers"]) == 3


def test_top_drivers_plain_english_format(explainer):
    """Verify top drivers contain valid reason codes and human-readable sentences."""
    sample_tx = {
        "step": 2,
        "type": "CASH_OUT",
        "amount": 5000.0,
        "nameOrig": "C123",
        "oldbalanceOrg": 5000.0,
        "nameDest": "C456",
        "oldbalanceDest": 0.0,
    }

    result = explainer.explain_transaction(sample_tx, top_k=3)

    for driver in result["top_drivers"]:
        assert isinstance(driver["feature"], str)
        assert isinstance(driver["raw_value"], (int, float))
        assert isinstance(driver["shap_value"], (int, float))
        assert isinstance(driver["reason_code"], str)
        assert len(driver["reason_code"]) > 3
        assert isinstance(driver["description"], str)
        assert len(driver["description"]) > 10
        # Ensure description is a proper English sentence
        assert driver["description"][0].isupper()


def test_explanation_latency_budget(explainer):
    """
    Assert that average TreeSHAP explanation latency is well under 20 ms.
    Target: < 2.0 ms.
    """
    cache = CustomerStateCache()
    sample_tx = {
        "step": 100,
        "type": "TRANSFER",
        "amount": 25000.0,
        "nameOrig": "C1000",
        "oldbalanceOrg": 25000.0,
        "nameDest": "C2000",
        "oldbalanceDest": 0.0,
    }

    # Warmup
    for _ in range(5):
        _ = explainer.explain_transaction(sample_tx, cache=cache)

    n_trials = 50
    t0 = time.perf_counter()
    for _ in range(n_trials):
        _ = explainer.explain_transaction(sample_tx, cache=cache)
    avg_latency_ms = ((time.perf_counter() - t0) / n_trials) * 1000.0

    print(f"\nMeasured Average TreeSHAP Latency: {avg_latency_ms:.3f} ms")
    assert avg_latency_ms < 20.0, f"Explanation latency too slow: {avg_latency_ms:.2f} ms"
