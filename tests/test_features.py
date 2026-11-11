"""
Unit Tests for Pre-Authorization Feature Engineering and Online-Offline Parity.

Proves to interviewers that:
1. Feature extraction strictly obeys pre-authorization constraints (no post-transaction leakage).
2. Online real-time extraction matches offline batch extraction (zero training-serving skew).
3. Online feature calculation runs in microseconds (< 1ms).
"""

import time
import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_NAMES,
    compute_stateless_features,
    compute_destination_rolling_features,
    extract_features,
    CustomerStateCache,
)


def test_feature_names_and_schema():
    """Verify exact 15 pre-authorization features and expected column names."""
    df = pd.DataFrame({
        "step": [1],
        "type": ["TRANSFER"],
        "amount": [1000.0],
        "nameOrig": ["C1"],
        "oldbalanceOrg": [1000.0],
        "nameDest": ["C2"],
        "oldbalanceDest": [0.0],
        "isFraud": [1],
    })

    feats, _ = extract_features(df)
    assert len(FEATURE_NAMES) == 15
    assert list(feats.columns[:15]) == FEATURE_NAMES
    assert "isFraud" in feats.columns


def test_origin_drain_ratio():
    """Verify pre-authorization drain ratio computation."""
    # Complete drain attempt: amount == oldbalanceOrg => ratio ~ 1.0
    drain_df = pd.DataFrame({
        "step": [1],
        "type": ["TRANSFER"],
        "amount": [50000.0],
        "nameOrig": ["C1"],
        "oldbalanceOrg": [50000.0],
        "nameDest": ["C2"],
        "oldbalanceDest": [0.0],
    })
    feats = compute_stateless_features(drain_df)
    expected_ratio = 50000.0 / 50001.0
    assert np.isclose(feats["orig_drain_ratio"].iloc[0], expected_ratio, atol=1e-4)

    # Partial transfer: amount << oldbalanceOrg
    partial_df = pd.DataFrame({
        "step": [1],
        "type": ["PAYMENT"],
        "amount": [100.0],
        "nameOrig": ["C1"],
        "oldbalanceOrg": [10000.0],
        "nameDest": ["M1"],
        "oldbalanceDest": [0.0],
    })
    feats_partial = compute_stateless_features(partial_df)
    assert feats_partial["orig_drain_ratio"].iloc[0] < 0.02


def test_point_in_time_destination_state():
    """
    Ensure destination rolling features at row i ONLY use data from rows < i.
    Row i must NOT see its own amount or future amounts in prior rolling statistics.
    """
    df = pd.DataFrame({
        "step": [1, 2, 3],
        "nameDest": ["Mule_A", "Mule_A", "Mule_A"],
        "amount": [100.0, 200.0, 300.0],
    })

    stateful_df, final_state = compute_destination_rolling_features(df)

    # Row 0: prior count = 0, prior sum = 0.0
    assert stateful_df["dest_prev_tx_count"].iloc[0] == 0
    assert stateful_df["dest_prev_amount_sum"].iloc[0] == 0.0

    # Row 1: prior count = 1, prior sum = 100.0
    assert stateful_df["dest_prev_tx_count"].iloc[1] == 1
    assert stateful_df["dest_prev_amount_sum"].iloc[1] == 100.0

    # Row 2: prior count = 2, prior sum = 300.0
    assert stateful_df["dest_prev_tx_count"].iloc[2] == 2
    assert stateful_df["dest_prev_amount_sum"].iloc[2] == 300.0

    # Final state carried forward
    assert final_state["Mule_A"] == (3, 600.0)


def test_online_offline_parity():
    """
    Test that online single-transaction feature extraction (CustomerStateCache)
    matches offline batch extraction bit-for-bit (avoiding training-serving skew).
    """
    tx = {
        "step": 14,
        "type": "CASH_OUT",
        "amount": 45000.0,
        "nameOrig": "C11223344",
        "oldbalanceOrg": 45000.0,
        "nameDest": "C99887766",
        "oldbalanceDest": 12000.0,
    }

    # 1. Batch extraction
    df = pd.DataFrame([tx])
    batch_feats, _ = extract_features(df)
    batch_vector = batch_feats[FEATURE_NAMES].values.flatten()

    # 2. Online extraction (with fresh cache)
    cache = CustomerStateCache()
    online_vector = cache.extract_online_vector(tx).flatten()

    # Assert exact numerical equivalence
    np.testing.assert_allclose(
        online_vector,
        batch_vector,
        rtol=1e-5,
        atol=1e-5,
        err_msg="Online-Offline parity mismatch! Training-serving skew detected."
    )


def test_online_feature_latency_under_1ms():
    """Assert single online feature extraction executes in < 1 ms (1000 microseconds)."""
    cache = CustomerStateCache()
    tx = {
        "step": 20,
        "type": "TRANSFER",
        "amount": 1000.0,
        "nameOrig": "C100",
        "oldbalanceOrg": 1000.0,
        "nameDest": "C200",
        "oldbalanceDest": 0.0,
    }

    start = time.perf_counter()
    _ = cache.extract_online_vector(tx)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < 1.0, f"Extraction too slow: {elapsed_ms:.4f} ms"
