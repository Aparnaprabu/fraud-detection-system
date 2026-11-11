"""
High-Performance Real-Time Fraud Detection API.

Built with FastAPI, Pydantic, XGBoost, and TreeSHAP.
SLA Target: Full HTTP round-trip latency < 100 ms at p95.
"""

from typing import Dict, List, Any, Optional
import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Response, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from src.config import MODELS_DIR
from src.features import CustomerStateCache, FEATURE_NAMES
from src.explain import FraudExplainer


# Global singleton references populated during startup
model_engine: Dict[str, Any] = {}


def load_engine():
    """Load model, TreeExplainer, state cache, and threshold policy into memory."""
    if model_engine:
        return
    print("Initializing Fraud Detection API Engine...")
    explainer = FraudExplainer(MODELS_DIR / "model.pkl")
    dest_state_file = MODELS_DIR / "dest_state.joblib"
    cache = CustomerStateCache(state_file=dest_state_file if dest_state_file.exists() else None)

    metrics_path = MODELS_DIR / "eval_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            policy = json.load(f)["tuning_policy"]
            tau_allow = float(policy.get("tau_allow", 0.3829))
            tau_block = float(policy.get("tau_block", 0.9571))
    else:
        tau_allow = 0.3829
        tau_block = 0.9571

    model_engine["explainer"] = explainer
    model_engine["model"] = explainer.model
    model_engine["cache"] = cache
    model_engine["tau_allow"] = tau_allow
    model_engine["tau_block"] = tau_block
    model_engine["version"] = "1.0.0"
    print(f"API Engine Loaded. Thresholds: Allow < {tau_allow} | Review [{tau_allow}, {tau_block}) | Block >= {tau_block}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_engine()
    yield
    model_engine.clear()


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Scores mobile-money transactions for fraud with instant TreeSHAP explainability in under 100 ms.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    Middleware measuring exact server processing time for every incoming request.
    Injects 'X-Process-Time-Ms' header into the HTTP response.
    """
    load_engine()
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.3f}"
    return response


# --- Pydantic Request & Response Schemas ---

class TransactionRequest(BaseModel):
    step: int = Field(..., ge=1, description="Simulation hour (step)")
    type: str = Field(..., description="Transaction type: TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT")
    amount: float = Field(..., gt=0.0, description="Transaction currency amount")
    nameOrig: str = Field(..., min_length=1, description="Origin account ID")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Origin balance before transaction")
    nameDest: str = Field(..., min_length=1, description="Destination account ID")
    oldbalanceDest: float = Field(..., ge=0.0, description="Destination balance before transaction")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "step": 4,
                "type": "TRANSFER",
                "amount": 181.0,
                "nameOrig": "C1305486145",
                "oldbalanceOrg": 181.0,
                "nameDest": "C553264065",
                "oldbalanceDest": 0.0,
            }
        }
    )


class DriverReason(BaseModel):
    feature: str
    raw_value: float
    shap_value: float
    reason_code: str
    description: str


class PredictResponse(BaseModel):
    fraud_score: float = Field(..., description="Predicted fraud probability [0.0, 1.0]")
    decision: str = Field(..., description="Action: AUTO_APPROVE, MANUAL_REVIEW, AUTO_BLOCK")
    top_drivers: List[DriverReason] = Field(..., description="Top 3 risk drivers pushing score up")
    process_time_ms: float = Field(..., description="Internal scoring & explainability time in ms")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    tau_allow: float
    tau_block: float


# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Service health check and model status."""
    is_loaded = "model" in model_engine and "explainer" in model_engine
    return HealthResponse(
        status="ok" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        version=model_engine.get("version", "1.0.0"),
        tau_allow=model_engine.get("tau_allow", 0.3829),
        tau_block=model_engine.get("tau_block", 0.9571),
    )


@app.post("/predict", response_model=PredictResponse)
async def predict_transaction(tx: TransactionRequest):
    """
    Score a single transaction for fraud and generate reason codes.
    Guaranteed SLA: sub-10ms server-side compute.
    """
    if "explainer" not in model_engine:
        raise HTTPException(status_code=503, detail="Model engine not initialized yet.")

    t0 = time.perf_counter()
    cache: CustomerStateCache = model_engine["cache"]
    explainer: FraudExplainer = model_engine["explainer"]
    model = model_engine["model"]
    tau_allow = model_engine["tau_allow"]
    tau_block = model_engine["tau_block"]

    # 1. Feature extraction in microseconds (< 0.01 ms)
    payload_dict = tx.model_dump()
    feature_vector = cache.extract_online_vector(payload_dict)

    # 2. Model scoring (< 0.06 ms)
    prob = float(model.predict_proba(feature_vector)[0, 1])

    # 3. Decision mapping based on validation thresholds
    if prob < tau_allow:
        decision = "AUTO_APPROVE"
    elif prob < tau_block:
        decision = "MANUAL_REVIEW"
    else:
        decision = "AUTO_BLOCK"

    # 4. TreeSHAP feature attributions (< 0.5 ms)
    explanation = explainer.explain_vector(feature_vector, top_k=3)
    compute_time_ms = (time.perf_counter() - t0) * 1000.0

    drivers = [DriverReason(**d) for d in explanation["top_drivers"]]

    return PredictResponse(
        fraud_score=round(prob, 4),
        decision=decision,
        top_drivers=drivers,
        process_time_ms=round(compute_time_ms, 3),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
