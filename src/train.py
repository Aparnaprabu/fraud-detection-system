"""
Model Training, Threshold Optimization, and Evaluation Module.

Engineering & Interview Concepts:
1. Class Imbalance Strategy: Cost-sensitive learning using 'scale_pos_weight' penalizes
   false negatives (missed frauds) without distorting threshold flexibility.
2. Threshold Tuning: In imbalanced fraud detection, the default 0.5 decision threshold
   is suboptimal. We sweep the Precision-Recall curve on the VALIDATION set to find
   an optimal threshold tau* maximizing F1 / catching >= 80% recall with high precision.
3. Out-of-Time Generalization: The test set is evaluated strictly using the threshold
   selected from validation, simulating genuine out-of-time production performance.
4. Tri-Tier Decision Policy:
   - Score < tau_allow       => AUTO_APPROVE
   - tau_allow <= Score < tau_block => MANUAL_REVIEW (Analyst queue / Step-up auth)
   - Score >= tau_block      => AUTO_BLOCK
"""

from typing import Dict, Any, Tuple, Optional
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)

from src.config import PROCESSED_DATA_DIR, MODELS_DIR, TARGET_COL
from src.features import FEATURE_NAMES


def load_feature_splits() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load precomputed feature sets for Train, Val, and Test partitions.
    """
    train_path = PROCESSED_DATA_DIR / "train_features.parquet"
    val_path = PROCESSED_DATA_DIR / "val_features.parquet"
    test_path = PROCESSED_DATA_DIR / "test_features.parquet"

    if not (train_path.exists() and val_path.exists() and test_path.exists()):
        raise FileNotFoundError(
            "Feature parquet files not found in data/processed/. Run src.features first."
        )

    print("Loading feature partitions from Parquet...")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    X_train = train_df[FEATURE_NAMES]
    y_train = train_df[TARGET_COL]

    X_val = val_df[FEATURE_NAMES]
    y_val = val_df[TARGET_COL]

    X_test = test_df[FEATURE_NAMES]
    y_test = test_df[TARGET_COL]

    print(f"  Train: {X_train.shape[0]:,} samples | {y_train.sum():,} frauds ({y_train.mean()*100:.3f}%)")
    print(f"  Val  : {X_val.shape[0]:,} samples | {y_val.sum():,} frauds ({y_val.mean()*100:.3f}%)")
    print(f"  Test : {X_test.shape[0]:,} samples | {y_test.sum():,} frauds ({y_test.mean()*100:.3f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_logistic_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[LogisticRegression, StandardScaler]:
    """Train a simple linear baseline with balanced class weights."""
    print("\n--- Training Baseline 1: Logistic Regression ---")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=200,
        random_state=42,
    )
    model.fit(X_scaled, y_train)
    return model, scaler


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float = 10.0,
) -> LGBMClassifier:
    """Train gradient boosted decision trees with LightGBM."""
    print(f"\n--- Training Champion 1: LightGBM (scale_pos_weight={scale_pos_weight}) ---")
    t0 = time.time()
    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    print(f"LightGBM trained in {time.time() - t0:.2f} seconds.")
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float = 10.0,
) -> XGBClassifier:
    """Train gradient boosted decision trees with XGBoost (histogram tree method)."""
    print(f"\n--- Training Champion 2: XGBoost (scale_pos_weight={scale_pos_weight}) ---")
    t0 = time.time()
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    print(f"XGBoost trained in {time.time() - t0:.2f} seconds.")
    return model


def find_optimal_threshold(
    y_true: pd.Series,
    y_prob: np.ndarray,
    min_recall: float = 0.80,
) -> Dict[str, Any]:
    """
    Tune the decision threshold tau* on the VALIDATION set.

    Finds the threshold that achieves target recall (>= 80%) with maximum precision,
    and calculates three-tier operational thresholds:
    - tau_allow: Threshold below which transactions are auto-approved
    - tau_block: Threshold above which transactions are immediately blocked
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Compute F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Filter thresholds meeting minimum recall criteria
    valid_indices = np.where(recalls[:-1] >= min_recall)[0]

    if len(valid_indices) > 0:
        # Choose the threshold that maximizes Precision among those with Recall >= min_recall
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
    else:
        # Fallback to maximum F1 threshold
        best_idx = np.argmax(f1_scores[:-1])

    best_threshold = float(thresholds[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])
    best_f1 = float(f1_scores[best_idx])

    # Tri-tier operational thresholds
    # Auto-allow threshold: captures <= 0.05 probability
    tau_allow = max(0.10, best_threshold * 0.4)
    # Auto-block threshold: our optimized threshold
    tau_block = best_threshold

    return {
        "best_threshold": round(best_threshold, 4),
        "val_precision_at_threshold": round(best_precision, 4),
        "val_recall_at_threshold": round(best_recall, 4),
        "val_f1_at_threshold": round(best_f1, 4),
        "tau_allow": round(tau_allow, 4),
        "tau_block": round(tau_block, 4),
    }


def evaluate_model_performance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    split_name: str = "Test",
    scaler: Optional[StandardScaler] = None,
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics on a dataset partition."""
    X_eval = scaler.transform(X) if scaler is not None else X

    probs = model.predict_proba(X_eval)[:, 1]
    preds = (probs >= threshold).astype(int)

    pr_auc = float(average_precision_score(y, probs))
    roc_auc = float(roc_auc_score(y, probs))

    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(f1_score(y, preds))

    results = {
        "split": split_name,
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    return results


def save_artifacts(
    champion_model: Any,
    all_metrics: Dict[str, Any],
    models_dir: Path = MODELS_DIR,
) -> None:
    """Serialize champion model and evaluation report to disk."""
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "model.pkl"
    metrics_path = models_dir / "eval_metrics.json"

    print(f"\nSaving champion model to: {model_path}")
    joblib.dump(champion_model, model_path)

    print(f"Saving evaluation report to: {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)


def main():
    """Execute end-to-end model training, threshold tuning, and test evaluation."""
    print("=" * 70)
    print("WEEK 3: MODEL TRAINING, COST-SENSITIVE LEARNING & PR-AUC EVALUATION")
    print("=" * 70)

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_feature_splits()

    # 2. Train Models
    # A) Linear Baseline
    lr_model, scaler = train_logistic_baseline(X_train, y_train)

    # B) Champion 1: LightGBM
    lgb_model = train_lightgbm(X_train, y_train, scale_pos_weight=10.0)

    # C) Champion 2: XGBoost
    xgb_model = train_xgboost(X_train, y_train, scale_pos_weight=10.0)

    # 3. Validation Tuning (Selecting Best Model & Optimal Threshold)
    print("\n" + "-" * 70)
    print("VALIDATION PR-AUC EVALUATION & THRESHOLD TUNING")
    print("-" * 70)

    # Evaluate validation PR-AUCs
    lr_val_probs = lr_model.predict_proba(scaler.transform(X_val))[:, 1]
    lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]

    lr_val_prauc = average_precision_score(y_val, lr_val_probs)
    lgb_val_prauc = average_precision_score(y_val, lgb_val_probs)
    xgb_val_prauc = average_precision_score(y_val, xgb_val_probs)

    print(f"  Logistic Regression Val PR-AUC : {lr_val_prauc:.4f}")
    print(f"  LightGBM            Val PR-AUC : {lgb_val_prauc:.4f}")
    print(f"  XGBoost             Val PR-AUC : {xgb_val_prauc:.4f}")

    # Tune threshold on champion (XGBoost)
    tuning_results = find_optimal_threshold(y_val, xgb_val_probs, min_recall=0.80)
    best_tau = tuning_results["best_threshold"]
    print(f"\nOptimal Decision Threshold tuned on Validation (tau*): {best_tau}")
    print(f"  Validation Precision at tau* : {tuning_results['val_precision_at_threshold']*100:.2f}%")
    print(f"  Validation Recall at tau*    : {tuning_results['val_recall_at_threshold']*100:.2f}%")
    print(f"  Three-Tier Decision Policy   : Allow < {tuning_results['tau_allow']} | Review [{tuning_results['tau_allow']}, {tuning_results['tau_block']}) | Block >= {tuning_results['tau_block']}")

    # 4. Out-of-Time Test Set Evaluation
    print("\n" + "-" * 70)
    print("OUT-OF-TIME GENERALIZATION EVALUATION (TEST SET)")
    print("-" * 70)

    lr_test_results = evaluate_model_performance(lr_model, X_test, y_test, threshold=0.5, split_name="Test (LR)", scaler=scaler)
    lgb_test_results = evaluate_model_performance(lgb_model, X_test, y_test, threshold=0.6731, split_name="Test (LightGBM)")
    xgb_test_results = evaluate_model_performance(xgb_model, X_test, y_test, threshold=best_tau, split_name="Test (XGBoost)")

    for name, res in [("Logistic Regression", lr_test_results), ("LightGBM", lgb_test_results), ("XGBoost", xgb_test_results)]:
        cm = res["confusion_matrix"]
        print(f"\n{name} Performance on Test Set:")
        print(f"  PR-AUC   : {res['pr_auc']:.4f}")
        print(f"  ROC-AUC  : {res['roc_auc']:.4f}")
        print(f"  Precision: {res['precision']*100:.2f}%")
        print(f"  Recall   : {res['recall']*100:.2f}%")
        print(f"  F1-Score : {res['f1_score']:.4f}")
        print(f"  Confusion: TP={cm['true_positives']}, FP={cm['false_positives']}, FN={cm['false_negatives']}, TN={cm['true_negatives']}")

    # 5. Save Champion Model & Evaluation Artifacts
    # Save XGBoost as champion and LightGBM as alternate
    all_metrics = {
        "champion_model": "XGBoost",
        "scale_pos_weight": 10.0,
        "tuning_policy": tuning_results,
        "models_comparison": {
            "logistic_regression": lr_test_results,
            "lightgbm": lgb_test_results,
            "xgboost": xgb_test_results,
        },
        "feature_importances": {
            k: float(v) for k, v in zip(FEATURE_NAMES, xgb_model.feature_importances_)
        },
    }

    save_artifacts(xgb_model, all_metrics)
    joblib.dump(lgb_model, MODELS_DIR / "lgbm_model.pkl")

    print("=" * 70)
    print("WEEK 3 MODEL PIPELINE COMPLETE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
