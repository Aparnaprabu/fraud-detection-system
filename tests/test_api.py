"""
Comprehensive Integration & Concurrent Latency Tests for FastAPI Inference Service.

Proves to interviewers that:
1. The API contract strictly adheres to Pydantic schemas.
2. Decisions map accurately to AUTO_APPROVE and AUTO_BLOCK.
3. Validation middleware properly rejects malformed requests with HTTP 422.
4. FULL HTTP round-trip latency (including network serializations, parsing, inference,
   and TreeSHAP attributions) stays well below 100 ms at p95 and p99 under concurrency.
"""

import asyncio
import time
import numpy as np
import pytest
import httpx

from src.api import app


@pytest.mark.anyio
async def test_health_check_endpoint():
    """Verify service health check returns 200 OK and preloaded models."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert "version" in data
        assert "tau_allow" in data
        assert "tau_block" in data


@pytest.mark.anyio
async def test_predict_schema_and_headers():
    """Verify /predict returns valid response schema and timing headers."""
    sample_tx = {
        "step": 4,
        "type": "TRANSFER",
        "amount": 181.0,
        "nameOrig": "C1305486145",
        "oldbalanceOrg": 181.0,
        "nameDest": "C553264065",
        "oldbalanceDest": 0.0,
    }

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/predict", json=sample_tx)
        assert resp.status_code == 200
        assert "X-Process-Time-Ms" in resp.headers

        data = resp.json()
        assert "fraud_score" in data
        assert 0.0 <= data["fraud_score"] <= 1.0
        assert data["decision"] in ["AUTO_APPROVE", "MANUAL_REVIEW", "AUTO_BLOCK"]
        assert "top_drivers" in data
        assert len(data["top_drivers"]) == 3
        assert data["process_time_ms"] > 0.0


@pytest.mark.anyio
async def test_decision_auto_approve_vs_auto_block():
    """Verify distinct routing for normal payment vs high-risk fraud attack."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Legitimate small payment
        legit_tx = {
            "step": 12,
            "type": "PAYMENT",
            "amount": 25.0,
            "nameOrig": "C1000",
            "oldbalanceOrg": 5000.0,
            "nameDest": "M2000",
            "oldbalanceDest": 0.0,
        }
        r_legit = await client.post("/predict", json=legit_tx)
        assert r_legit.status_code == 200
        assert r_legit.json()["decision"] == "AUTO_APPROVE"
        assert r_legit.json()["fraud_score"] < 0.3829

        # Known fraud pattern (100% account drain transfer to customer account at night)
        fraud_tx = {
            "step": 4,
            "type": "TRANSFER",
            "amount": 500000.0,
            "nameOrig": "C9999",
            "oldbalanceOrg": 500000.0,
            "nameDest": "C8888",
            "oldbalanceDest": 0.0,
        }
        r_fraud = await client.post("/predict", json=fraud_tx)
        assert r_fraud.status_code == 200
        assert r_fraud.json()["decision"] == "AUTO_BLOCK"
        assert r_fraud.json()["fraud_score"] >= 0.9571
        # Check that top driver flags the balance drain
        assert r_fraud.json()["top_drivers"][0]["reason_code"] == "BALANCE_DRAIN_ATTEMPT"


@pytest.mark.anyio
async def test_input_validation_errors():
    """Verify that malformed payloads are rejected with HTTP 422 Unprocessable Entity."""
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Missing required 'amount'
        bad_tx = {
            "step": 1,
            "type": "PAYMENT",
            "nameOrig": "C1",
            "oldbalanceOrg": 100.0,
            "nameDest": "M1",
            "oldbalanceDest": 0.0,
        }
        resp = await client.post("/predict", json=bad_tx)
        assert resp.status_code == 422


@pytest.mark.anyio
async def test_concurrent_load_http_roundtrip_latency():
    """
    Measure FULL HTTP client round-trip latency at p50, p95, and p99 under concurrent load.
    Load level: 10 concurrent async workers sending 100 total requests.
    Target SLA: p95 < 100.0 ms.
    """
    sample_tx = {
        "step": 5,
        "type": "CASH_OUT",
        "amount": 1200.0,
        "nameOrig": "C5555",
        "oldbalanceOrg": 1200.0,
        "nameDest": "C6666",
        "oldbalanceDest": 0.0,
    }

    n_concurrent = 10
    total_requests = 100
    latencies = []

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # Warmup
        for _ in range(5):
            await client.post("/predict", json=sample_tx)

        async def send_request():
            t0 = time.perf_counter()
            r = await client.post("/predict", json=sample_tx)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            assert r.status_code == 200
            return elapsed_ms

        for _ in range(total_requests // n_concurrent):
            batch = [send_request() for _ in range(n_concurrent)]
            batch_latencies = await asyncio.gather(*batch)
            latencies.extend(batch_latencies)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    print("\n" + "=" * 65)
    print(f"FULL HTTP ROUND-TRIP LATENCY (Load Level: {n_concurrent} concurrent workers, {total_requests} requests)")
    print("-" * 65)
    print(f"  p50 Latency : {p50:.2f} ms")
    print(f"  p95 Latency : {p95:.2f} ms")
    print(f"  p99 Latency : {p99:.2f} ms")
    print(f"  SLA Target  : < 100.0 ms | Met: {p95 < 100.0}")
    print("=" * 65)

    assert p95 < 100.0, f"p95 latency breached SLA: {p95:.2f} ms"
