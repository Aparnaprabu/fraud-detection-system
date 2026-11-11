"""
Real-Time Transaction Stream Simulator.

Simulates production payment traffic by sequentially replaying transactions
from the holdout Test set (`steps 651–743`) and streaming them to the FastAPI
inference service.
"""

from typing import Optional, Dict, Any, Generator
import time
import argparse
import requests
import pandas as pd
from src.config import PROCESSED_DATA_DIR


DEFAULT_API_URL = "http://127.0.0.1:8000/predict"


def load_test_transactions(max_records: Optional[int] = None) -> pd.DataFrame:
    """Load chronologically ordered test transactions."""
    test_path = PROCESSED_DATA_DIR / "test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"Test partition not found at '{test_path}'. Run src.data_loader first.")

    df = pd.read_parquet(test_path)
    if max_records:
        df = df.iloc[:max_records]
    return df


def generate_transaction_stream(
    df: pd.DataFrame,
    interleave_fraud_ratio: Optional[float] = None,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yields transactions in chronological order.
    Optionally enriches stream with a higher proportion of fraud events for demo purposes.
    """
    if interleave_fraud_ratio is not None and 0.0 < interleave_fraud_ratio < 1.0:
        frauds = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0]
        # Interleave to showcase both allowed payments and blocked attacks
        f_idx = 0
        l_idx = 0
        while l_idx < len(legit) or f_idx < len(frauds):
            if (f_idx < len(frauds)) and (np.random.random() < interleave_fraud_ratio or l_idx >= len(legit)):
                row = frauds.iloc[f_idx]
                f_idx += 1
            else:
                row = legit.iloc[l_idx]
                l_idx += 1

            yield {
                "step": int(row["step"]),
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "nameOrig": str(row["nameOrig"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "nameDest": str(row["nameDest"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "ground_truth_fraud": int(row["isFraud"]),
            }
    else:
        for _, row in df.iterrows():
            yield {
                "step": int(row["step"]),
                "type": str(row["type"]),
                "amount": float(row["amount"]),
                "nameOrig": str(row["nameOrig"]),
                "oldbalanceOrg": float(row["oldbalanceOrg"]),
                "nameDest": str(row["nameDest"]),
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "ground_truth_fraud": int(row["isFraud"]),
            }


def stream_to_api(
    api_url: str = DEFAULT_API_URL,
    n_transactions: int = 50,
    delay_ms: float = 20.0,
    interleave_fraud: bool = True,
) -> Dict[str, Any]:
    """
    Stream live transactions sequentially to the API and record latency metrics.
    """
    print(f"Connecting to API at: {api_url}")
    print(f"Streaming {n_transactions} transactions with {delay_ms} ms delay...")

    df = load_test_transactions()
    stream = generate_transaction_stream(df, interleave_fraud_ratio=0.3 if interleave_fraud else None)

    stats = {
        "total": 0,
        "approved": 0,
        "review": 0,
        "blocked": 0,
        "latencies": [],
    }

    session = requests.Session()

    print("\n" + "=" * 90)
    print(f"{'STEP':<6} | {'TYPE':<9} | {'AMOUNT':>12} | {'SCORE':>7} | {'DECISION':<13} | {'LATENCY':>8} | TOP REASON")
    print("-" * 90)

    for i, tx in enumerate(stream):
        if i >= n_transactions:
            break

        # Remove ground truth before posting to API
        payload = {k: v for k, v in tx.items() if k != "ground_truth_fraud"}

        t0 = time.perf_counter()
        try:
            resp = session.post(api_url, json=payload, timeout=2.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                score = data["fraud_score"]
                decision = data["decision"]
                top_reason = data["top_drivers"][0]["reason_code"] if data["top_drivers"] else "NONE"

                stats["total"] += 1
                stats["latencies"].append(elapsed_ms)

                if decision == "AUTO_APPROVE":
                    stats["approved"] += 1
                elif decision == "MANUAL_REVIEW":
                    stats["review"] += 1
                else:
                    stats["blocked"] += 1

                print(f"{payload['step']:<6} | {payload['type']:<9} | ${payload['amount']:>11,.2f} | {score*100:>6.1f}% | {decision:<13} | {elapsed_ms:>6.2f}ms | {top_reason}")
            else:
                print(f"API Error {resp.status_code}: {resp.text}")

        except Exception as e:
            print(f"Connection failed: {e}")
            break

        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)

    print("=" * 90)
    if stats["latencies"]:
        lats = stats["latencies"]
        print(f"\nStream Completed: {stats['total']} transactions sent.")
        print(f"  Approved       : {stats['approved']} ({stats['approved']/stats['total']*100:.1f}%)")
        print(f"  Manual Review  : {stats['review']} ({stats['review']/stats['total']*100:.1f}%)")
        print(f"  Blocked        : {stats['blocked']} ({stats['blocked']/stats['total']*100:.1f}%)")
        print(f"  Avg HTTP Latency: {sum(lats)/len(lats):.2f} ms")
        print(f"  p95 HTTP Latency: {np.percentile(lats, 95):.2f} ms")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Payment Transaction Simulator")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="FastAPI predict endpoint URL")
    parser.add_argument("--count", type=int, default=30, help="Number of transactions to stream")
    parser.add_argument("--delay", type=float, default=20.0, help="Delay between transactions in ms")
    args = parser.parse_args()

    stream_to_api(api_url=args.url, n_transactions=args.count, delay_ms=args.delay)


if __name__ == "__main__":
    import numpy as np
    main()
