"""
Pre-Authorization Point-in-Time Feature Engineering Module.

Key Engineering Rules:
1. Pre-Authorization Correctness: Features are computed strictly using data available
   BEFORE a transaction is authorized (request parameters, prior balances, historical aggregates).
2. Zero Target Leakage: Post-settlement balances ('newbalanceOrig', 'newbalanceDest')
   and their derived balance discrepancy flags are strictly excluded.
3. Dual Extraction Modes (Online-Offline Parity):
   - Batch Mode: Vectorized processing across historical partitions.
   - Online Mode: In-memory cache (CustomerStateCache) providing sub-millisecond
     feature extraction for the real-time FastAPI inference service.
"""

from typing import Dict, Tuple, Optional, Any, List
import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    TARGET_COL,
    TIME_COL,
)

# Canonical list of 15 clean pre-authorization features
FEATURE_NAMES: List[str] = [
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "hour_of_day",
    "orig_drain_ratio",
    "is_dest_zero_before",
    "is_merchant_dest",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_PAYMENT",
    "type_CASH_IN",
    "type_DEBIT",
    "dest_prev_tx_count",
    "dest_prev_amount_sum",
    "dest_prev_amount_mean",
]

TRANSACTION_TYPES = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]


def compute_stateless_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pre-authorization features independent of historical context.

    Features include:
    - Raw request amounts and ledger balances before clearing
    - Diurnal pattern: hour of day (step % 24)
    - Origin drain ratio: amount / (oldbalanceOrg + 1.0)
    - Destination zero balance before transfer
    - Merchant destination flag (nameDest starts with 'M')
    - One-hot binary columns for transaction types
    """
    feats = pd.DataFrame(index=df.index)

    # 1. Pre-clearing ledger balances and requested amount
    feats["amount"] = df["amount"].astype("float64")
    feats["oldbalanceOrg"] = df["oldbalanceOrg"].astype("float64")
    feats["oldbalanceDest"] = df["oldbalanceDest"].astype("float64")

    # 2. Temporal diurnal feature (0-23 hours)
    feats["hour_of_day"] = (df[TIME_COL] % 24).astype("uint8")

    # 3. Behavioral ratio: checks if transaction attempts to drain 100% of origin balance
    feats["orig_drain_ratio"] = feats["amount"] / (feats["oldbalanceOrg"] + 1.0)

    # 4. Account state flags prior to clearing
    feats["is_dest_zero_before"] = (feats["oldbalanceDest"] == 0.0).astype("uint8")
    feats["is_merchant_dest"] = df["nameDest"].str.startswith("M").astype("uint8")

    # 5. One-hot encoded transaction types
    for tx_type in TRANSACTION_TYPES:
        feats[f"type_{tx_type}"] = (df["type"] == tx_type).astype("uint8")

    return feats


def compute_destination_rolling_features(
    df: pd.DataFrame,
    initial_state: Optional[Dict[str, Tuple[int, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, float]]]:
    """
    Compute point-in-time stateful features for destination accounts (nameDest).

    Each transaction row at time t only incorporates transactions received
    by that recipient strictly prior to row t.
    """
    stateful_df = pd.DataFrame(index=df.index)

    # Cumulative counts strictly prior to current row within partition
    internal_cumcount = df.groupby("nameDest", observed=True).cumcount().astype("uint32")
    internal_cumsum = (
        df.groupby("nameDest", observed=True)["amount"].cumsum() - df["amount"]
    ).astype("float64")

    # If historical state from prior split is supplied, offset the cumulative values
    if initial_state:
        prior_counts = {k: v[0] for k, v in initial_state.items()}
        prior_sums = {k: v[1] for k, v in initial_state.items()}

        offset_count = df["nameDest"].map(prior_counts).fillna(0).astype("uint32")
        offset_sum = df["nameDest"].map(prior_sums).fillna(0.0).astype("float64")

        total_prev_count = internal_cumcount + offset_count
        total_prev_sum = internal_cumsum + offset_sum
    else:
        total_prev_count = internal_cumcount
        total_prev_sum = internal_cumsum

    stateful_df["dest_prev_tx_count"] = total_prev_count.astype("uint32")
    stateful_df["dest_prev_amount_sum"] = total_prev_sum.astype("float64")
    stateful_df["dest_prev_amount_mean"] = (
        total_prev_sum / (total_prev_count + 1.0)
    ).astype("float64")

    # Compute updated final state to pass forward
    final_state = dict(initial_state) if initial_state else {}
    summary = df.groupby("nameDest", observed=True)["amount"].agg(count="count", total="sum")

    for name, row in summary.iterrows():
        prev_c, prev_s = final_state.get(name, (0, 0.0))
        final_state[name] = (prev_c + int(row["count"]), prev_s + float(row["total"]))

    return stateful_df, final_state


def extract_features(
    df: pd.DataFrame,
    initial_state: Optional[Dict[str, Tuple[int, float]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[int, float]]]:
    """
    Extract complete 15-dimensional clean pre-authorization feature set.
    """
    stateless_df = compute_stateless_features(df)
    stateful_df, final_state = compute_destination_rolling_features(df, initial_state)

    # Combine stateless and stateful features
    features_df = pd.concat([stateless_df, stateful_df], axis=1)

    # Enforce exact canonical column ordering
    features_df = features_df[FEATURE_NAMES]

    # If ground-truth label exists, preserve it for supervised modeling
    if TARGET_COL in df.columns:
        features_df[TARGET_COL] = df[TARGET_COL].values

    return features_df, final_state


class CustomerStateCache:
    """
    High-performance in-memory state cache for real-time online inference in FastAPI.
    """

    def __init__(
        self,
        preloaded_state: Optional[Dict[str, Tuple[int, float]]] = None,
        state_file: Optional[Path] = None,
    ):
        self._state: Dict[str, List[float]] = {}
        if state_file and Path(state_file).exists():
            import joblib
            raw = joblib.load(state_file)
            for k, (cnt, sm) in raw.items():
                self._state[k] = [float(cnt), float(sm)]
        elif preloaded_state:
            for k, (cnt, sm) in preloaded_state.items():
                self._state[k] = [float(cnt), float(sm)]

    def get_destination_history(self, name_dest: str) -> Tuple[int, float, float]:
        """Query prior transaction count, prior amount sum, and prior mean."""
        entry = self._state.get(name_dest)
        if entry is None:
            return 0, 0.0, 0.0
        prev_count = int(entry[0])
        prev_sum = entry[1]
        prev_mean = prev_sum / (prev_count + 1.0)
        return prev_count, prev_sum, prev_mean

    def update_destination(self, name_dest: str, amount: float) -> None:
        """Update destination account state upon transaction processing."""
        entry = self._state.get(name_dest)
        if entry is None:
            self._state[name_dest] = [1.0, float(amount)]
        else:
            entry[0] += 1.0
            entry[1] += float(amount)

    def extract_online_vector(self, tx: Dict[str, Any]) -> np.ndarray:
        """
        Convert a single raw transaction dictionary into a 1x15 numpy feature array.
        Pre-authorization: only requires request fields and pre-clearing balances.
        """
        step = int(tx.get("step", 1))
        tx_type = str(tx.get("type", "TRANSFER")).upper()
        amount = float(tx.get("amount", 0.0))
        oldbalanceOrg = float(tx.get("oldbalanceOrg", 0.0))
        nameDest = str(tx.get("nameDest", ""))
        oldbalanceDest = float(tx.get("oldbalanceDest", 0.0))

        # Stateless pre-auth calculations
        hour_of_day = step % 24
        orig_drain_ratio = amount / (oldbalanceOrg + 1.0)
        is_dest_zero_before = 1.0 if oldbalanceDest == 0.0 else 0.0
        is_merchant_dest = 1.0 if nameDest.startswith("M") else 0.0

        # Type one-hot flags
        type_flags = [1.0 if tx_type == t else 0.0 for t in TRANSACTION_TYPES]

        # Stateful destination features
        prev_count, prev_sum, prev_mean = self.get_destination_history(nameDest)

        # Assemble fixed 15-dimensional feature vector
        vector = np.array([
            amount,
            oldbalanceOrg,
            oldbalanceDest,
            hour_of_day,
            orig_drain_ratio,
            is_dest_zero_before,
            is_merchant_dest,
            type_flags[0],  # TRANSFER
            type_flags[1],  # CASH_OUT
            type_flags[2],  # PAYMENT
            type_flags[3],  # CASH_IN
            type_flags[4],  # DEBIT
            prev_count,
            prev_sum,
            prev_mean,
        ], dtype=np.float64)

        # Increment state for future transactions
        self.update_destination(nameDest, amount)

        return vector.reshape(1, -1)


def save_feature_metadata(
    feature_df: pd.DataFrame,
    output_path: Path = MODELS_DIR / "feature_meta.json",
) -> None:
    """Save clean feature schema, ordering, and baseline stats to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    numeric_cols = [c for c in FEATURE_NAMES if c in feature_df.columns]
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "mean": float(feature_df[col].mean()),
            "std": float(feature_df[col].std()),
            "min": float(feature_df[col].min()),
            "max": float(feature_df[col].max()),
        }

    meta = {
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "baseline_stats": stats,
    }

    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved feature metadata ({len(FEATURE_NAMES)} clean features) to: {output_path}")


def main():
    """Run batch clean feature extraction across Train, Val, and Test splits."""
    print("=" * 70)
    print("PRE-AUTHORIZATION FEATURE ENGINEERING PIPELINE (NO TARGET LEAKAGE)")
    print("=" * 70)

    from src.data_loader import load_splits
    train_df, val_df, test_df = load_splits()

    print("\nExtracting clean features for Train set (steps 1–550)...")
    train_feats, train_state = extract_features(train_df, initial_state=None)

    print("\nExtracting clean features for Validation set (steps 551–650)...")
    val_feats, val_state = extract_features(val_df, initial_state=train_state)

    print("\nExtracting clean features for Test set (steps 651–743)...")
    test_feats, test_state = extract_features(test_df, initial_state=val_state)

    # Save clean feature parquet files
    print("\nSaving clean feature partitions to Parquet...")
    train_feats.to_parquet(PROCESSED_DATA_DIR / "train_features.parquet", index=False, engine="pyarrow")
    val_feats.to_parquet(PROCESSED_DATA_DIR / "val_features.parquet", index=False, engine="pyarrow")
    test_feats.to_parquet(PROCESSED_DATA_DIR / "test_features.parquet", index=False, engine="pyarrow")

    save_feature_metadata(train_feats)

    print(f"Train features: {train_feats.shape}, Val: {val_feats.shape}, Test: {test_feats.shape}")
    print("=" * 70)
    print("CLEAN FEATURE PIPELINE COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
