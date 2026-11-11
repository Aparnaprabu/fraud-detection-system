"""
Reusable Visualization Components for Fraud Monitoring Dashboard.
Uses Plotly for interactive risk gauges and SHAP contribution charts.
"""

from typing import List, Dict, Any
import plotly.graph_objects as go


def create_risk_gauge(
    score: float,
    tau_allow: float = 0.3829,
    tau_block: float = 0.9571,
) -> go.Figure:
    """
    Renders a Plotly gauge chart displaying fraud risk probability with tri-tier zones:
    - Green  [0.0, tau_allow)       => AUTO_APPROVE
    - Yellow [tau_allow, tau_block) => MANUAL_REVIEW
    - Red    [tau_block, 1.0]       => AUTO_BLOCK
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 36, "color": "#1f2937"}},
            title={"text": "<b>Fraud Risk Probability</b>", "font": {"size": 18, "color": "#374151"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#9ca3af"},
                "bar": {"color": "#dc2626" if score >= tau_block else ("#d97706" if score >= tau_allow else "#16a34a"), "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#e5e7eb",
                "steps": [
                    {"range": [0, tau_allow * 100], "color": "rgba(22, 163, 74, 0.15)"},
                    {"range": [tau_allow * 100, tau_block * 100], "color": "rgba(217, 119, 6, 0.20)"},
                    {"range": [tau_block * 100, 100], "color": "rgba(220, 38, 38, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 4},
                    "thickness": 0.8,
                    "value": score * 100,
                },
            },
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif"),
    )
    return fig


def create_shap_bar_chart(drivers: List[Dict[str, Any]]) -> go.Figure:
    """
    Renders a horizontal bar chart displaying the top risk factors pushing the score up.
    """
    if not drivers:
        fig = go.Figure()
        fig.add_annotation(text="No positive risk drivers for this transaction.", showarrow=False)
        return fig

    # Reverse so top factor is at the top
    features = [d["feature"].replace("_", " ").title() for d in reversed(drivers)]
    shap_vals = [d["shap_value"] for d in reversed(drivers)]
    raw_vals = [f"Raw: {d['raw_value']:.2f}" for d in reversed(drivers)]

    fig = go.Figure(
        go.Bar(
            x=shap_vals,
            y=features,
            orientation="h",
            text=raw_vals,
            textposition="auto",
            marker=dict(
                color="rgba(220, 38, 38, 0.85)",
                line=dict(color="#991b1b", width=1.5),
            ),
        )
    )

    fig.update_layout(
        title=dict(text="<b>Top Risk Factor Attributions (TreeSHAP)</b>", font=dict(size=16)),
        xaxis_title="SHAP Attribution (Pushing Risk UP)",
        yaxis_title="",
        height=260,
        margin=dict(l=20, r=20, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(243, 244, 246, 0.5)",
    )
    return fig
