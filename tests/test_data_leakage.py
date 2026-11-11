"""
Unit Tests for Zero Temporal Data Leakage.

These tests prove to interviewers that our pipeline enforces strict chronological
separation, ensuring no future data leaks into the training or validation sets.
"""

import pandas as pd
import pytest
from src.config import PROCESSED_DATA_DIR, TARGET_COL, TIME_COL
from src.data_loader import split_data_by_time, load_splits


def test_split_logic_on_synthetic_data():
    """
    Test split_data_by_time logic on a deterministic synthetic DataFrame.
    """
    dummy_data = pd.DataFrame({
        "step": [10, 50, 100, 550, 551, 600, 650, 651, 700],
        "isFraud": [0, 0, 1, 0, 1, 0, 0, 1, 0],
        "amount": [100.0] * 9,
    })

    train, val, test = split_data_by_time(dummy_data, train_step_end=550, val_step_end=650)

    # Invariants:
    # 1. Monotonic step constraints
    assert train["step"].max() <= 550
    assert val["step"].min() >= 551
    assert val["step"].max() <= 650
    assert test["step"].min() >= 651

    # 2. Strict chronological ordering between sets
    assert train["step"].max() < val["step"].min()
    assert val["step"].max() < test["step"].min()

    # 3. Row conservation
    assert len(train) + len(val) + len(test) == len(dummy_data)


def test_processed_parquet_leakage_invariants():
    """
    Verify temporal separation directly on the actual processed Parquet datasets.
    """
    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"

    # Skip if parquet files have not been generated yet
    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        pytest.skip("Processed parquet splits not generated yet. Run src.data_loader first.")

    train_df, val_df, test_df = load_splits()

    # 1. Temporal boundary invariant: max(train) < min(val) < min(test)
    max_train_step = train_df[TIME_COL].max()
    min_val_step = val_df[TIME_COL].min()
    max_val_step = val_df[TIME_COL].max()
    min_test_step = test_df[TIME_COL].min()

    assert max_train_step < min_val_step, (
        f"Temporal Leak! Train max step ({max_train_step}) >= Val min step ({min_val_step})"
    )
    assert max_val_step < min_test_step, (
        f"Temporal Leak! Val max step ({max_val_step}) >= Test min step ({min_test_step})"
    )

    # 2. Positive class presence: Every partition must contain frauds for valid evaluation
    assert train_df[TARGET_COL].sum() > 0, "Train set has no fraud records!"
    assert val_df[TARGET_COL].sum() > 0, "Validation set has no fraud records!"
    assert test_df[TARGET_COL].sum() > 0, "Test set has no fraud records!"
