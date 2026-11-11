"""
Data Ingestion, Profiling, and Chronological Splitting Module.

Key Engineering Rules:
1. NEVER split randomly: Random k-fold or train_test_split causes temporal data
   leakage in fraud/time-series systems (future fraud patterns leak into the past).
2. Data is partitioned strictly by the 'step' column (1 step = 1 hour).
3. Memory optimization: Downcast integer types and convert categories to minimize RAM footprint.
4. Outputs are saved as compressed Parquet files for fast I/O and strict schema preservation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_DIR,
    TRAIN_STEP_END,
    VAL_STEP_END,
    TARGET_COL,
    TIME_COL,
)

# Optimized dtypes to reduce RAM usage from ~600MB to ~250MB
OPTIMIZED_DTYPES = {
    "step": "uint16",
    "type": "category",
    "amount": "float64",
    "nameOrig": "string",
    "oldbalanceOrg": "float64",
    "newbalanceOrig": "float64",
    "nameDest": "string",
    "oldbalanceDest": "float64",
    "newbalanceDest": "float64",
    "isFraud": "uint8",
    "isFlaggedFraud": "uint8",
}


def load_raw_data(filepath: Path = RAW_DATA_FILE, nrows: int = None) -> pd.DataFrame:
    """
    Load raw PaySim CSV dataset with memory-optimized data types.

    Parameters
    ----------
    filepath : Path
        Path to the raw PaySim CSV file.
    nrows : int, optional
        Number of rows to read (useful for testing and fast prototyping).

    Returns
    -------
    pd.DataFrame
        Loaded PaySim dataframe.
    """
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please place the PaySim CSV in data/raw/."
        )

    print(f"Loading raw data from: {filepath.name} ...")
    df = pd.read_csv(filepath, dtype=OPTIMIZED_DTYPES, nrows=nrows)
    print(f"Loaded {len(df):,} transactions successfully.")
    return df


def profile_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute exploratory data analysis (EDA) statistics on the PaySim dataset.

    Returns key insights required for ML system design and technical interviews:
    - Overall fraud prevalence (extreme class imbalance)
    - Fraud distribution by transaction type (TRANSFER vs CASH_OUT)
    - Rule-based baseline evaluation (isFlaggedFraud heuristic performance)
    - Temporal range (steps)

    Parameters
    ----------
    df : pd.DataFrame
        The loaded PaySim dataframe.

    Returns
    -------
    dict
        Structured summary dictionary containing profiling metrics.
    """
    total_rows = len(df)
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    min_step = int(df[TIME_COL].min())
    max_step = int(df[TIME_COL].max())

    total_frauds = int(df[TARGET_COL].sum())
    fraud_rate_pct = (total_frauds / total_rows) * 100

    # Fraud distribution across transaction types
    type_group = df.groupby("type", observed=True)[TARGET_COL].agg(
        total_count="count", fraud_count="sum"
    )
    type_group["fraud_rate_pct"] = (
        type_group["fraud_count"] / type_group["total_count"]
    ) * 100

    # Rule-based heuristic analysis (isFlaggedFraud flags transactions > 200,000)
    rule_flagged = int(df["isFlaggedFraud"].sum())
    rule_true_positives = int(
        df[(df["isFlaggedFraud"] == 1) & (df[TARGET_COL] == 1)].shape[0]
    )
    rule_precision = (
        (rule_true_positives / rule_flagged * 100) if rule_flagged > 0 else 0.0
    )
    rule_recall = (
        (rule_true_positives / total_frauds * 100) if total_frauds > 0 else 0.0
    )

    # Missing value check
    null_counts = df.isnull().sum().to_dict()

    summary = {
        "total_rows": total_rows,
        "memory_mb": round(memory_mb, 2),
        "step_range": (min_step, max_step),
        "total_frauds": total_frauds,
        "fraud_rate_pct": round(fraud_rate_pct, 4),
        "type_breakdown": type_group.to_dict(orient="index"),
        "rule_heuristic": {
            "flagged_count": rule_flagged,
            "true_positives": rule_true_positives,
            "precision_pct": round(rule_precision, 2),
            "recall_pct": round(rule_recall, 4),
        },
        "null_counts": null_counts,
    }

    return summary


def split_data_by_time(
    df: pd.DataFrame,
    train_step_end: int = TRAIN_STEP_END,
    val_step_end: int = VAL_STEP_END,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset chronologically based on the 'step' column.

    Strict Temporal Invariant:
        max(train_step) < min(val_step) <= max(val_step) < min(test_step)

    Parameters
    ----------
    df : pd.DataFrame
        Complete PaySim dataframe.
    train_step_end : int
        Last step for the training set (default: 550).
    val_step_end : int
        Last step for the validation set (default: 650).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    print(f"Splitting data chronologically: Train (<= {train_step_end}), "
          f"Val ({train_step_end + 1} to {val_step_end}), Test (> {val_step_end}) ...")

    train_mask = df[TIME_COL] <= train_step_end
    val_mask = (df[TIME_COL] > train_step_end) & (df[TIME_COL] <= val_step_end)
    test_mask = df[TIME_COL] > val_step_end

    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    test_df = df.loc[test_mask].reset_index(drop=True)

    # Sanity check: Ensure strict temporal boundary without leakage
    assert train_df[TIME_COL].max() < val_df[TIME_COL].min(), "Leakage detected: Train overlaps with Validation!"
    assert val_df[TIME_COL].max() < test_df[TIME_COL].min(), "Leakage detected: Validation overlaps with Test!"

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Dict[str, Path]:
    """
    Save train, validation, and test sets as compressed Parquet files.

    Why Parquet?
    1. Preserves column data types (categories, floats, uints) exactly.
    2. Columnar compression reduces disk space significantly compared to raw CSV.
    3. Faster read speeds during model training and pipeline runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": output_dir / "train.parquet",
        "val": output_dir / "val.parquet",
        "test": output_dir / "test.parquet",
    }

    print("Saving processed partitions to Parquet...")
    train_df.to_parquet(paths["train"], index=False, engine="pyarrow")
    val_df.to_parquet(paths["val"], index=False, engine="pyarrow")
    test_df.to_parquet(paths["test"], index=False, engine="pyarrow")
    print(f"Successfully saved all partitions to: {output_dir}")

    return paths


def load_splits(processed_dir: Path = PROCESSED_DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load previously saved Parquet splits.
    """
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            f"Processed parquet files not found in '{processed_dir}'. Run data_loader first."
        )

    train_df = pd.read_parquet(train_path, engine="pyarrow")
    val_df = pd.read_parquet(val_path, engine="pyarrow")
    test_df = pd.read_parquet(test_path, engine="pyarrow")

    return train_df, val_df, test_df


def main():
    """Run end-to-end data ingestion, profiling, and chronological splitting."""
    print("=" * 70)
    print("WEEK 1: DATA INGESTION & CHRONOLOGICAL LEAK-FREE SPLITTING")
    print("=" * 70)

    # 1. Load data
    df = load_raw_data()

    # 2. Profile data
    print("\n" + "-" * 70)
    print("DATA PROFILING & EXPLORATORY METRICS")
    print("-" * 70)
    profile = profile_dataset(df)

    print(f"Total Transactions : {profile['total_rows']:,}")
    print(f"RAM Memory Footprint: {profile['memory_mb']:.1f} MB (after optimization)")
    print(f"Temporal Steps     : {profile['step_range'][0]} to {profile['step_range'][1]} hours")
    print(f"Total Frauds       : {profile['total_frauds']:,} ({profile['fraud_rate_pct']:.4f}%)")

    print("\nFraud Breakdown by Transaction Type:")
    for tx_type, stats in profile["type_breakdown"].items():
        print(f"  - {tx_type:<10}: {stats['total_count']:>9,} total | "
              f"{stats['fraud_count']:>5,} frauds ({stats['fraud_rate_pct']:>6.3f}%)")

    print("\nExisting Rule Heuristic (isFlaggedFraud):")
    rule = profile["rule_heuristic"]
    print(f"  - Transactions Flagged : {rule['flagged_count']}")
    print(f"  - True Frauds Caught   : {rule['true_positives']}")
    print(f"  - Rule Precision       : {rule['precision_pct']:.2f}%")
    print(f"  - Rule Recall          : {rule['recall_pct']:.4f}% (Misses >99.8% of frauds!)")

    # 3. Split by time
    print("\n" + "-" * 70)
    print("CHRONOLOGICAL TEMPORAL SPLITTING")
    print("-" * 70)
    train_df, val_df, test_df = split_data_by_time(df)

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        frauds = split[TARGET_COL].sum()
        pct = (frauds / len(split)) * 100
        step_min = split[TIME_COL].min()
        step_max = split[TIME_COL].max()
        print(f"  {name:<5}: {len(split):>9,} rows (steps {step_min:>3}–{step_max:<3}) | "
              f"{frauds:>5,} frauds ({pct:.3f}%)")

    # 4. Save splits
    print("\n" + "-" * 70)
    save_splits(train_df, val_df, test_df)
    print("=" * 70)
    print("WEEK 1 DATA PIPELINE COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
