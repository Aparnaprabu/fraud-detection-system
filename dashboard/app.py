"""
Real-Time Fraud Detection & Alert Monitoring Dashboard.
Built with Streamlit and Plotly.
"""

import os
import time
from typing import Dict, Any, List
import requests
import pandas as pd
import numpy as np
import streamlit as st

from dashboard.components import create_risk_gauge, create_shap_bar_chart


# Set page configuration
st.set_page_config(
    page_title="Real-Time Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API Endpoint URL (configurable via environment variable for Docker)
DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health(api_base_url: str) -> Dict[str, Any]:
    """Query API /health endpoint to check service status."""
    try:
        resp = requests.get(f"{api_base_url}/health", timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return {"status": "offline", "model_loaded": False}


def score_transaction_api(api_base_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send transaction to API /predict endpoint."""
    resp = requests.post(f"{api_base_url}/predict", json=payload, timeout=3.0)
    if resp.status_code == 200:
        return resp.json()
    raise RuntimeError(f"API Error {resp.status_code}: {resp.text}")


# --- Initialize Session State ---
if "streamed_transactions" not in st.session_state:
    st.session_state.streamed_transactions = []
if "selected_transaction" not in st.session_state:
    st.session_state.selected_transaction = None


# --- Sidebar ---
st.sidebar.title("🛡️ Fraud Engine")
st.sidebar.caption("Pre-Authorization Decision System")

api_url = st.sidebar.text_input("API Base URL", value=DEFAULT_API_URL)
health = check_api_health(api_url)

if health.get("status") == "ok":
    st.sidebar.success(f"🟢 Connected to API (v{health.get('version', '1.0.0')})")
    tau_allow = health.get("tau_allow", 0.3829)
    tau_block = health.get("tau_block", 0.9571)
else:
    st.sidebar.error("🔴 API Offline - Start FastAPI server on port 8000")
    tau_allow = 0.3829
    tau_block = 0.9571

st.sidebar.markdown("---")
st.sidebar.subheader("Tri-Tier Decision Policy")
st.sidebar.markdown(f"""
- 🟢 **AUTO_APPROVE**: Score < `{tau_allow:.4f}`
- 🟡 **MANUAL_REVIEW**: `{tau_allow:.4f}` ≤ Score < `{tau_block:.4f}`
- 🔴 **AUTO_BLOCK**: Score ≥ `{tau_block:.4f}`
""")

st.sidebar.markdown("---")
st.sidebar.caption("System Stack: XGBoost (15 Pre-Auth Features) + TreeSHAP + FastAPI")


# --- Main Header ---
st.title("🛡️ Real-Time Payment Fraud Monitoring System")
st.markdown(
    "Pre-authorization payment scoring, automated risk routing, and instant TreeSHAP reason code generation in under 100 ms."
)


# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Streaming Feed",
    "🚨 Operations Triage Queue",
    "🔍 SHAP Transaction Inspector",
    "🧪 Manual Testing Sandbox",
])


# ==============================================================================
# TAB 1: LIVE STREAMING FEED
# ==============================================================================
with tab1:
    st.subheader("Live Payment Transaction Stream")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 4])

    with col_ctrl1:
        stream_count = st.number_input("Batch Size", min_value=5, max_value=200, value=25, step=5)
    with col_ctrl2:
        stream_delay = st.slider("Stream Interval (ms)", min_value=0, max_value=200, value=30, step=10)
    with col_ctrl3:
        st.write("")
        st.write("")
        btn_start_stream = st.button("▶️ Stream Transactions from Test Set", type="primary", use_container_width=True)

    if btn_start_stream:
        if health.get("status") != "ok":
            st.error("Cannot stream: FastAPI service is offline. Please start the API.")
        else:
            # Load real test partition transactions
            try:
                from src.simulator import load_test_transactions, generate_transaction_stream
                test_df = load_test_transactions()
                stream_gen = generate_transaction_stream(test_df, interleave_fraud_ratio=0.35)
            except Exception as e:
                st.warning(f"Using synthetic demo transactions: {e}")
                # Fallback generator
                def demo_gen():
                    while True:
                        yield {
                            "step": np.random.randint(651, 743),
                            "type": np.random.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], p=[0.3, 0.3, 0.4]),
                            "amount": round(np.random.exponential(15000), 2),
                            "nameOrig": f"C{np.random.randint(100000, 999999)}",
                            "oldbalanceOrg": round(np.random.exponential(20000), 2),
                            "nameDest": f"C{np.random.randint(100000, 999999)}",
                            "oldbalanceDest": 0.0,
                        }
                stream_gen = demo_gen()

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(stream_count):
                raw_tx = next(stream_gen)
                payload = {k: v for k, v in raw_tx.items() if k != "ground_truth_fraud"}

                t_start = time.perf_counter()
                try:
                    result = score_transaction_api(api_url, payload)
                    rtt_ms = (time.perf_counter() - t_start) * 1000.0

                    record = {
                        "tx_id": f"TX-{len(st.session_state.streamed_transactions) + 1:05d}",
                        "step": payload["step"],
                        "type": payload["type"],
                        "amount": payload["amount"],
                        "nameOrig": payload["nameOrig"],
                        "nameDest": payload["nameDest"],
                        "fraud_score": result["fraud_score"],
                        "decision": result["decision"],
                        "top_reason": result["top_drivers"][0]["reason_code"] if result["top_drivers"] else "NONE",
                        "top_drivers": result["top_drivers"],
                        "rtt_latency_ms": round(rtt_ms, 2),
                        "server_time_ms": result["process_time_ms"],
                    }
                    st.session_state.streamed_transactions.insert(0, record)
                except Exception as e:
                    st.error(f"Failed scoring: {e}")
                    break

                progress_bar.progress((i + 1) / stream_count)
                status_text.text(f"Processed {i + 1}/{stream_count} transactions...")
                if stream_delay > 0:
                    time.sleep(stream_delay / 1000.0)

            status_text.text(f"Stream complete: {stream_count} transactions scored.")

    # KPI Summary Cards
    if st.session_state.streamed_transactions:
        tx_data = st.session_state.streamed_transactions
        total_tx = len(tx_data)
        approved_tx = sum(1 for t in tx_data if t["decision"] == "AUTO_APPROVE")
        review_tx = sum(1 for t in tx_data if t["decision"] == "MANUAL_REVIEW")
        blocked_tx = sum(1 for t in tx_data if t["decision"] == "AUTO_BLOCK")
        avg_rtt = sum(t["rtt_latency_ms"] for t in tx_data) / total_tx
        p95_rtt = np.percentile([t["rtt_latency_ms"] for t in tx_data], 95)

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Scored", f"{total_tx:,}")
        kpi2.metric("🟢 Auto-Approved", f"{approved_tx:,}", f"{(approved_tx/total_tx)*100:.1f}%")
        kpi3.metric("🟡 Manual Review", f"{review_tx:,}", f"{(review_tx/total_tx)*100:.1f}%")
        kpi4.metric("🔴 Auto-Blocked", f"{blocked_tx:,}", f"{(blocked_tx/total_tx)*100:.1f}%")
        kpi5.metric("⚡ p95 Latency", f"{p95_rtt:.1f} ms", f"avg {avg_rtt:.1f} ms")

        st.markdown("### Recent Transactions")
        df_display = pd.DataFrame(tx_data)[
            ["tx_id", "step", "type", "amount", "fraud_score", "decision", "top_reason", "rtt_latency_ms"]
        ]
        df_display["amount"] = df_display["amount"].apply(lambda x: f"${x:,.2f}")
        df_display["fraud_score"] = df_display["fraud_score"].apply(lambda x: f"{x*100:.1f}%")
        df_display["rtt_latency_ms"] = df_display["rtt_latency_ms"].apply(lambda x: f"{x:.1f} ms")

        st.dataframe(df_display, use_container_width=True, height=350)
    else:
        st.info("No transactions streamed yet. Click **▶️ Stream Transactions from Test Set** above to start streaming.")


# ==============================================================================
# TAB 2: OPERATIONS TRIAGE QUEUE
# ==============================================================================
with tab2:
    st.subheader("Operations Alert Triage Queue")
    st.caption("Investigator queue containing transactions flagged for Review or Block.")

    if st.session_state.streamed_transactions:
        filter_opt = st.radio(
            "Filter Queue By Severity",
            ["All Flagged (Review + Block)", "🔴 AUTO_BLOCK Only", "🟡 MANUAL_REVIEW Only"],
            horizontal=True,
        )

        all_tx = st.session_state.streamed_transactions
        if filter_opt == "🔴 AUTO_BLOCK Only":
            filtered_tx = [t for t in all_tx if t["decision"] == "AUTO_BLOCK"]
        elif filter_opt == "🟡 MANUAL_REVIEW Only":
            filtered_tx = [t for t in all_tx if t["decision"] == "MANUAL_REVIEW"]
        else:
            filtered_tx = [t for t in all_tx if t["decision"] in ["AUTO_BLOCK", "MANUAL_REVIEW"]]

        st.markdown(f"**{len(filtered_tx)} transactions** requiring investigator attention.")

        if filtered_tx:
            triage_df = pd.DataFrame(filtered_tx)[
                ["tx_id", "type", "amount", "fraud_score", "decision", "top_reason", "rtt_latency_ms"]
            ]
            triage_df["amount"] = triage_df["amount"].apply(lambda x: f"${x:,.2f}")
            triage_df["fraud_score"] = triage_df["fraud_score"].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(triage_df, use_container_width=True, height=280)

            # Quick inspect selector
            st.markdown("#### Select Transaction for Deep-Dive Investigation")
            tx_options = [t["tx_id"] for t in filtered_tx]
            selected_tx_id = st.selectbox("Choose Transaction ID:", tx_options)

            if st.button("Inspect in SHAP View 🔍"):
                match = next(t for t in filtered_tx if t["tx_id"] == selected_tx_id)
                st.session_state.selected_transaction = match
                st.success(f"Loaded {selected_tx_id}. Open the **🔍 SHAP Transaction Inspector** tab to view details!")
        else:
            st.success("Triage queue empty! No transactions match the current filter.")
    else:
        st.info("Run the transaction stream in Tab 1 to populate the triage queue.")


# ==============================================================================
# TAB 3: SHAP TRANSACTION INSPECTOR
# ==============================================================================
with tab3:
    st.subheader("Deep-Dive SHAP Explanation Inspector")
    st.caption("Local feature attributions and compliance reason codes for flagged transactions.")

    tx = st.session_state.selected_transaction
    if tx is None and st.session_state.streamed_transactions:
        # Default to first non-approved transaction
        flagged = [t for t in st.session_state.streamed_transactions if t["decision"] != "AUTO_APPROVE"]
        tx = flagged[0] if flagged else st.session_state.streamed_transactions[0]

    if tx:
        st.markdown(f"### Investigating: `{tx.get('tx_id', 'TX-UNKNOWN')}`")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(create_risk_gauge(tx["fraud_score"], tau_allow, tau_block), use_container_width=True)

            st.markdown("#### Transaction Parameters")
            st.json({
                "Transaction ID": tx.get("tx_id"),
                "Type": tx.get("type"),
                "Amount": f"${tx.get('amount', 0):,.2f}",
                "Sender": tx.get("nameOrig"),
                "Recipient": tx.get("nameDest"),
                "Decision": tx.get("decision"),
                "Fraud Score": f"{tx.get('fraud_score')*100:.2f}%",
                "HTTP Latency": f"{tx.get('rtt_latency_ms')} ms",
            })

        with c2:
            st.plotly_chart(create_shap_bar_chart(tx.get("top_drivers", [])), use_container_width=True)

            st.markdown("#### Reason Code Breakdown")
            for i, driver in enumerate(tx.get("top_drivers", []), 1):
                with st.expander(f"{i}. [{driver['reason_code']}] — SHAP: +{driver['shap_value']:.4f}", expanded=True):
                    st.write(f"**Explanation:** {driver['description']}")
                    st.caption(f"Feature: `{driver['feature']}` | Raw Value: `{driver['raw_value']}`")
    else:
        st.info("No transaction selected. Stream transactions in Tab 1 or pick one from Tab 2.")


# ==============================================================================
# TAB 4: MANUAL TESTING SANDBOX
# ==============================================================================
with tab4:
    st.subheader("Manual Transaction Scoring Sandbox")
    st.caption("Craft custom payment requests to test live fraud decisioning and TreeSHAP explainability.")

    with st.form("manual_tx_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            m_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
            m_amount = st.number_input("Amount ($)", min_value=1.0, value=25000.0, step=100.0)
            m_step = st.slider("Simulation Hour (Step)", min_value=1, max_value=744, value=4)
            m_nameOrig = st.text_input("Origin Account ID", value="C123456789")

        with col_f2:
            m_oldbalanceOrg = st.number_input("Sender Ledger Balance ($)", min_value=0.0, value=25000.0, step=100.0)
            m_nameDest = st.text_input("Recipient Account ID", value="C987654321")
            m_oldbalanceDest = st.number_input("Recipient Ledger Balance ($)", min_value=0.0, value=0.0, step=100.0)

        submit_btn = st.form_submit_button("⚡ Score Transaction in Real-Time", type="primary", use_container_width=True)

    if submit_btn:
        payload = {
            "step": m_step,
            "type": m_type,
            "amount": m_amount,
            "nameOrig": m_nameOrig,
            "oldbalanceOrg": m_oldbalanceOrg,
            "nameDest": m_nameDest,
            "oldbalanceDest": m_oldbalanceDest,
        }

        try:
            t_start = time.perf_counter()
            result = score_transaction_api(api_url, payload)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            st.markdown("---")
            st.subheader("Scoring Result")

            res_c1, res_c2 = st.columns([1, 1])
            with res_c1:
                st.plotly_chart(create_risk_gauge(result["fraud_score"], tau_allow, tau_block), use_container_width=True)
                dec = result["decision"]
                if dec == "AUTO_APPROVE":
                    st.success(f"### Decision: 🟢 {dec}")
                elif dec == "MANUAL_REVIEW":
                    st.warning(f"### Decision: 🟡 {dec}")
                else:
                    st.error(f"### Decision: 🔴 {dec}")
                st.metric("Round-Trip Latency", f"{elapsed_ms:.2f} ms", f"Engine: {result['process_time_ms']:.2f} ms")

            with res_c2:
                st.plotly_chart(create_shap_bar_chart(result["top_drivers"]), use_container_width=True)
                st.markdown("#### Primary Reason Codes")
                for i, d in enumerate(result["top_drivers"], 1):
                    st.info(f"**{i}. {d['reason_code']}**: {d['description']}")

        except Exception as e:
            st.error(f"Scoring request failed: {e}")
