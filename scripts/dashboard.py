"""Eco-Loop Dashboard — run with:  streamlit run scripts/dashboard.py

Shows:
  - Cumulative kWh: baseline vs agent (the money shot)
  - Peak demand comparison
  - Zone temperature + setpoint trace
  - PMV distribution vs ASHRAE 55 comfort band
  - LLM decision timeline with reasoning text
  - Summary results table
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from ecoloop import config as C

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Eco-Loop Building Agent",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 Eco-Loop Building Agent")
st.caption("Autonomous closed-loop building control via EnergyPlus + LLM + MCP")

# ── Load data ─────────────────────────────────────────────────────────────────
def load_timeseries(mode: str) -> pd.DataFrame | None:
    p = C.RESULTS / mode / "timeseries.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)

def load_summary(mode: str) -> dict | None:
    p = C.RESULTS / mode / "summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())

def load_decisions() -> list:
    p = C.RESULTS / "agent" / "decisions.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())

base_df = load_timeseries("baseline")
agent_df = load_timeseries("agent")
base_sum = load_summary("baseline")
agent_sum = load_summary("agent")
decisions = load_decisions()

# ── Check data available ──────────────────────────────────────────────────────
if base_df is None and agent_df is None:
    st.warning("No simulation data found yet.")
    st.info("""
**To generate data, run:**
```
python scripts/run_agent.py
```
Or for a quick static test:
```
python scripts/run.py --both --cooling-offset 0 --unocc-cooling-offset 0.5
```
    """)
    st.stop()

# ── Summary cards ─────────────────────────────────────────────────────────────
if base_sum and agent_sum:
    st.subheader("Results Summary")
    c1, c2, c3, c4, c5 = st.columns(5)

    kwh_delta = ((agent_sum["kwh_total"] - base_sum["kwh_total"])
                 / base_sum["kwh_total"] * 100)
    peak_delta = ((agent_sum["peak_kw"] - base_sum["peak_kw"])
                  / base_sum["peak_kw"] * 100)
    carbon_delta = ((agent_sum["kgco2"] - base_sum["kgco2"])
                    / base_sum["kgco2"] * 100)
    pmv_base = base_sum["pmv_compliance"] * 100
    pmv_agent = agent_sum["pmv_compliance"] * 100

    c1.metric("Total kWh", f"{agent_sum['kwh_total']:,.0f}",
              f"{kwh_delta:+.1f}% vs baseline",
              delta_color="inverse")
    c2.metric("Peak Demand", f"{agent_sum['peak_kw']:.1f} kW",
              f"{peak_delta:+.1f}% vs baseline",
              delta_color="inverse")
    c3.metric("Carbon", f"{agent_sum['kgco2']:,.0f} kgCO₂",
              f"{carbon_delta:+.1f}% vs baseline",
              delta_color="inverse")
    c4.metric("PMV Compliance", f"{pmv_agent:.1f}%",
              f"{pmv_agent - pmv_base:+.1f}% vs baseline",
              delta_color="normal")
    c5.metric("LLM Decisions", str(agent_sum.get("decisions", len(decisions))))

    comfort_ok = pmv_agent >= pmv_base - 2.0
    if comfort_ok:
        st.success("✅ Comfort maintained — energy savings are valid")
    else:
        st.error("⚠️ Comfort degraded vs baseline")

st.divider()

# ── Chart 1: Cumulative kWh ───────────────────────────────────────────────────
st.subheader("⚡ Cumulative Electricity — Baseline vs Agent")

fig_kwh = go.Figure()

if base_df is not None:
    base_df["cumkwh"] = base_df["kwh_total"].cumsum()
    base_df["label"] = (base_df["day"].astype(str) + "-" +
                        base_df["hour"].round(1).astype(str))
    fig_kwh.add_trace(go.Scatter(
        x=list(range(len(base_df))),
        y=base_df["cumkwh"],
        name="Baseline",
        line=dict(color="#ef4444", width=2, dash="dot"),
    ))

if agent_df is not None:
    agent_df["cumkwh"] = agent_df["kwh_total"].cumsum()
    fig_kwh.add_trace(go.Scatter(
        x=list(range(len(agent_df))),
        y=agent_df["cumkwh"],
        name="Agent",
        line=dict(color="#22c55e", width=2),
    ))

    # Annotate LLM decisions on the chart
    for d in decisions[:20]:  # max 20 annotations
        step = d.get("step", 0)
        if step < len(agent_df):
            y_val = agent_df["cumkwh"].iloc[step]
            reason = d.get("reason", "")[:40]
            fig_kwh.add_annotation(
                x=step, y=y_val,
                text="🤖", showarrow=True,
                arrowhead=2, arrowsize=1,
                arrowcolor="#a855f7",
                font=dict(size=14),
                hovertext=reason,
            )

fig_kwh.update_layout(
    xaxis_title="Timestep (15-min intervals)",
    yaxis_title="Cumulative kWh",
    legend=dict(orientation="h", y=1.1),
    height=350,
    margin=dict(t=20, b=40),
)
st.plotly_chart(fig_kwh, use_container_width=True)
st.caption("🤖 markers show when the LLM agent issued a new control policy")

# ── Chart 2: PMV Distribution ─────────────────────────────────────────────────
st.subheader("🌡️ Thermal Comfort (PMV) — Occupied Hours Only")

fig_pmv = go.Figure()

comfort_lo, comfort_hi = C.PMV_LOW, C.PMV_HIGH

if base_df is not None:
    occ_base = base_df[base_df["occupied"] == 1]["mean_pmv"].dropna()
    fig_pmv.add_trace(go.Histogram(
        x=occ_base, name="Baseline",
        opacity=0.6, marker_color="#ef4444",
        xbins=dict(size=0.1),
    ))

if agent_df is not None:
    occ_agent = agent_df[agent_df["occupied"] == 1]["mean_pmv"].dropna()
    fig_pmv.add_trace(go.Histogram(
        x=occ_agent, name="Agent",
        opacity=0.6, marker_color="#22c55e",
        xbins=dict(size=0.1),
    ))

# ASHRAE 55 comfort band
fig_pmv.add_vrect(x0=comfort_lo, x1=comfort_hi,
                  fillcolor="#22c55e", opacity=0.1,
                  annotation_text="ASHRAE 55 comfort band",
                  annotation_position="top left")

fig_pmv.update_layout(
    barmode="overlay",
    xaxis_title="PMV (Predicted Mean Vote)",
    yaxis_title="Count",
    height=300,
    margin=dict(t=20, b=40),
)
st.plotly_chart(fig_pmv, use_container_width=True)

# ── Chart 3: Setpoint trace ───────────────────────────────────────────────────
if agent_df is not None:
    st.subheader("🎛️ Cooling Setpoints — Scheduled vs Agent-Controlled")

    fig_sp = go.Figure()
    x = list(range(len(agent_df)))

    fig_sp.add_trace(go.Scatter(
        x=x, y=agent_df["base_cool_sp"],
        name="Scheduled (baseline)",
        line=dict(color="#94a3b8", width=1, dash="dot"),
    ))

    # Show occupied vs unoccupied periods as background
    occ_mask = agent_df["occupied"] == 1
    fig_sp.add_trace(go.Scatter(
        x=x, y=agent_df["mean_temp"],
        name="Mean zone temp",
        line=dict(color="#f97316", width=1),
        opacity=0.7,
    ))

    fig_sp.update_layout(
        xaxis_title="Timestep",
        yaxis_title="Temperature (°C)",
        height=280,
        margin=dict(t=20, b=40),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_sp, use_container_width=True)

# ── LLM Decision Log ─────────────────────────────────────────────────────────
st.subheader("🤖 LLM Agent Decision Log")

if decisions:
    for i, d in enumerate(reversed(decisions[-15:])):
        p = d.get("policy", {})
        reason = p.get("reason", d.get("reason", "—"))
        step = d.get("step", "?")
        with st.expander(f"Decision {len(decisions)-i}: step {step} — {reason[:60]}"):
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "cooling_offset": p.get("cooling_offset", 0),
                    "unocc_cooling_offset": p.get("unocc_cooling_offset", 0),
                    "precool_hours": p.get("precool_hours", 0),
                    "peak_offset": p.get("peak_offset", 0),
                })
            with col2:
                st.json({
                    "heating_offset": p.get("heating_offset", 0),
                    "unocc_heating_offset": p.get("unocc_heating_offset", 0),
                    "precool_depth": p.get("precool_depth", 0),
                    "peak_start": p.get("peak_start", 17),
                    "peak_end": p.get("peak_end", 21),
                })
            st.caption(f"Full reason: {reason}")
else:
    st.info("No agent decisions yet — run `python scripts/run_agent.py` first")

# ── Carbon Intensity Profile ──────────────────────────────────────────────────
st.subheader("⚡ Grid Carbon Intensity Profile")
hours = list(range(24))
carbon = C.CARBON_INTENSITY
fig_c = px.bar(x=hours, y=carbon,
               labels={"x": "Hour of day", "y": "gCO₂/kWh"},
               color=carbon,
               color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"])
fig_c.update_layout(height=250, margin=dict(t=10, b=40),
                    coloraxis_showscale=False)
fig_c.add_vrect(x0=C.OCCUPIED_START - 0.5, x1=C.OCCUPIED_END - 0.5,
                fillcolor="#3b82f6", opacity=0.1,
                annotation_text="occupied hours")
st.plotly_chart(fig_c, use_container_width=True)
st.caption("Agent precools during green (cheap/clean) hours and relaxes setpoints during red (expensive/dirty) hours")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Eco-Loop · EnergyPlus 24.1 · Python Runtime API · MCP · Qwen2.5-7B · "
    f"Building: DOE Medium Office · Weather: Tampa TMY3 · "
    f"Simulation: 14 days July · 4 timesteps/hour"
)
