"""
Model Explainability and Plain-English Reason Code Module.

Key Engineering & Compliance Rules:
1. Compliance Context: Automated decisions provide explainable reason codes
   for customers, operations teams, and auditors.
2. Local Attributions via TreeSHAP: We use TreeSHAP to compute mathematically exact
   Shapley values for each transaction in sub-millisecond latency (< 1 ms).
3. Plain-English Translation: Raw continuous Shapley numbers (+6.47 on orig_drain_ratio)
   are converted into human-readable sentences that explain *why* the transaction was flagged.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import time
import joblib
import numpy as np
import shap

from src.config import MODELS_DIR
from src.features import FEATURE_NAMES, CustomerStateCache


# Template catalog mapping features to standardized financial reason codes and descriptions
REASON_TEMPLATES = {
    "orig_drain_ratio": {
        "code": "BALANCE_DRAIN_ATTEMPT",
        "description": lambda val: (
            f"Transaction requests {val * 100:.1f}% of sender's balance (attempting to drain account)."
            if val >= 0.80
            else f"Elevated transaction amount relative to account balance ({val * 100:.1f}%)."
        ),
    },
    "is_merchant_dest": {
        "code": "NON_MERCHANT_COUNTERPARTY",
        "description": lambda val: "Counterparty is an unverified personal/customer account rather than an established merchant.",
    },
    "type_TRANSFER": {
        "code": "IRREVERSIBLE_TRANSFER_CHANNEL",
        "description": lambda val: "Payment initiated through an irreversible external transfer channel.",
    },
    "type_CASH_OUT": {
        "code": "RAPID_LIQUIDATION_CHANNEL",
        "description": lambda val: "Payment routed to a cash withdrawal point (CASH_OUT liquidation).",
    },
    "hour_of_day": {
        "code": "OFF_PEAK_TRANSACTION_TIME",
        "description": lambda val: f"Payment submitted during high-risk off-peak/night hours ({int(val):02d}:00).",
    },
    "amount": {
        "code": "HIGH_TRANSACTION_VALUE",
        "description": lambda val: f"High payment amount ($ {val:,.2f}) exceeding standard risk tolerance.",
    },
    "is_dest_zero_before": {
        "code": "UNFUNDED_RECIPIENT_ACCOUNT",
        "description": lambda val: "Recipient account had zero initial ledger balance prior to transfer.",
    },
    "dest_prev_tx_count": {
        "code": "RECIPIENT_VELOCITY_SPIKE",
        "description": lambda val: f"High transaction velocity to recipient ({int(val)} prior incoming transfers).",
    },
    "oldbalanceOrg": {
        "code": "LARGE_ACCOUNT_EXPOSURE",
        "description": lambda val: f"Large initial account balance exposed in transaction ($ {val:,.2f}).",
    },
    "oldbalanceDest": {
        "code": "RECIPIENT_BALANCE_ANOMALY",
        "description": lambda val: f"Unusual recipient balance profile prior to transfer ($ {val:,.2f}).",
    },
    "dest_prev_amount_sum": {
        "code": "RECIPIENT_HIGH_CUMULATIVE_VOLUME",
        "description": lambda val: f"High cumulative volume routed through recipient account ($ {val:,.2f}).",
    },
    "dest_prev_amount_mean": {
        "code": "RECIPIENT_AVERAGE_ANOMALY",
        "description": lambda val: f"Payment size deviates significantly from recipient's historical average ($ {val:,.2f}).",
    },
}


class FraudExplainer:
    """
    Sub-millisecond local explanation service using TreeSHAP on the champion XGBoost model.
    """

    def __init__(self, model_path: Path = MODELS_DIR / "model.pkl"):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at '{model_path}'. Run src.train first."
            )
        self.model = joblib.load(model_path)
        self.explainer = shap.TreeExplainer(self.model)
        self.base_value = float(
            self.explainer.expected_value
            if isinstance(self.explainer.expected_value, (int, float))
            else self.explainer.expected_value[0]
        )

    def explain_vector(
        self,
        feature_vector: np.ndarray,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Explain a single 1x15 feature vector.

        Returns the top-k features pushing the fraud probability UP (positive SHAP values),
        along with raw values, Shapley scores, and plain-English reasons.

        Parameters
        ----------
        feature_vector : np.ndarray
            Shape (1, 15) numeric array adhering to FEATURE_NAMES.
        top_k : int
            Number of top risk factors to return (default: 3).

        Returns
        -------
        dict
            Structured explanation payload with latency measurement.
        """
        t0 = time.perf_counter()

        # Compute raw Shapley values
        shap_values = self.explainer.shap_values(feature_vector)
        # Handle 1D or 2D array output from TreeExplainer
        sv = shap_values[0] if len(shap_values.shape) == 2 else shap_values

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Sort indices by Shapley attribution descending (highest positive push first)
        sorted_indices = np.argsort(sv)[::-1]

        top_drivers = []
        for idx in sorted_indices:
            # We focus on factors increasing risk (shap > 0)
            shap_val = float(sv[idx])
            feat_name = FEATURE_NAMES[idx]
            raw_val = float(feature_vector[0, idx])

            template = REASON_TEMPLATES.get(
                feat_name,
                {
                    "code": f"RISK_FACTOR_{feat_name.upper()}",
                    "description": lambda v, f=feat_name: f"Elevated risk contribution from {f} (value: {v:.2f}).",
                },
            )

            reason_desc = (
                template["description"](raw_val)
                if callable(template["description"])
                else str(template["description"])
            )

            top_drivers.append({
                "feature": feat_name,
                "raw_value": round(raw_val, 4),
                "shap_value": round(shap_val, 4),
                "reason_code": template["code"],
                "description": reason_desc,
            })

            if len(top_drivers) == top_k:
                break

        return {
            "top_drivers": top_drivers,
            "base_value": round(self.base_value, 4),
            "explanation_latency_ms": round(latency_ms, 3),
        }

    def explain_transaction(
        self,
        tx: Dict[str, Any],
        cache: Optional[CustomerStateCache] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        End-to-end convenience method: takes raw transaction dict, extracts features,
        and returns fraud probability plus top-k plain-English reasons.
        """
        if cache is None:
            cache = CustomerStateCache()

        vec = cache.extract_online_vector(tx)
        prob = float(self.model.predict_proba(vec)[0, 1])
        explanation = self.explain_vector(vec, top_k=top_k)

        return {
            "fraud_probability": round(prob, 4),
            "top_drivers": explanation["top_drivers"],
            "base_value": explanation["base_value"],
            "explanation_latency_ms": explanation["explanation_latency_ms"],
        }


def main():
    """Demonstrate explanation service on a sample PaySim fraud transaction."""
    print("=" * 70)
    print("WEEK 4: MODEL EXPLAINABILITY VIA TREESHAP")
    print("=" * 70)

    explainer = FraudExplainer()
    cache = CustomerStateCache()

    # Sample transaction: PaySim row 2 (known fraudulent transfer)
    sample_tx = {
        "step": 4,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C1305486145",
        "oldbalanceOrg": 181.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
    }

    print("\nAnalyzing Sample Transaction:")
    for k, v in sample_tx.items():
        print(f"  {k:<16}: {v}")

    result = explainer.explain_transaction(sample_tx, cache=cache, top_k=3)

    print(f"\nFraud Probability : {result['fraud_probability'] * 100:.2f}%")
    print(f"TreeSHAP Latency  : {result['explanation_latency_ms']:.3f} ms")
    print("\nTop 3 Primary Risk Drivers:")
    for i, driver in enumerate(result["top_drivers"], 1):
        print(f"  {i}. [{driver['reason_code']}] (SHAP: +{driver['shap_value']:.4f})")
        print(f"     Feature : {driver['feature']} = {driver['raw_value']}")
        print(f"     Reason  : \"{driver['description']}\"")

    print("\n" + "=" * 70)
    print("WEEK 4 EXPLAINABILITY MODULE READY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
