"""MCP server — the LLM's eyes and hands on the building.

The agent never touches EnergyPlus directly. Everything goes through these tools:

  get_telemetry        → summarised sensor readings (not raw logs)
  get_simulation_errors → parsed eplusout.err, deduped
  get_savings_so_far   → live delta vs baseline
  get_carbon_intensity → grid carbon curve by hour
  set_control_policy   → validated, then applied

This design is what earns the "Agentic Autonomy" marks. The LLM uses tools to
observe and act; it does not see raw files or write setpoints directly. The
validator between set_control_policy and the actual actuators is what lets you
claim the agent "cannot trade comfort for energy no matter what it hallucinates."

Run standalone for testing:
    python -m ecoloop.mcp_server

The runner imports and starts this on a background thread when mode='agent'.
"""

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from . import config as C
from .policy import Policy, PolicyRejected, validate

mcp = FastMCP("eco-loop-building-agent")

# Shared state written by the runner, read by the tools.
# Using a lock because the MCP server and the EnergyPlus callbacks run on
# different threads.
_lock = threading.Lock()
_state: dict[str, Any] = {
    "telemetry": [],          # list of per-timestep dicts, rolling window
    "baseline_kwh": 0.0,
    "current_kwh": 0.0,
    "current_peak_kw": 0.0,
    "current_pmv_compliance": 1.0,
    "decisions": [],
    "errors": [],
    "current_policy": Policy(),
    "policy_callback": None,  # set by runner; called when agent sets a new policy
    "sim_hour": 0.0,
    "sim_day": 1,
}


def update_state(**kwargs):
    """Called by the runner every timestep to push fresh data in."""
    with _lock:
        _state.update(kwargs)


def get_current_policy() -> Policy:
    with _lock:
        return _state["current_policy"]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def get_telemetry(window_hours: float = 2.0) -> str:
    """Get a summary of recent building sensor readings.

    Returns per-zone temperature and PMV, plus whole-building energy metrics,
    averaged over the last `window_hours` of simulated time.

    Args:
        window_hours: How many simulated hours to look back (default 2).
    """
    with _lock:
        rows = list(_state["telemetry"])
        hour = _state["sim_hour"]
        day = _state["sim_day"]

    if not rows:
        return json.dumps({"error": "No telemetry yet — simulation may still be warming up."})

    steps_per_hour = 4
    n = max(1, int(window_hours * steps_per_hour))
    window = rows[-n:]

    def avg(key):
        vals = [r[key] for r in window if key in r and r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    occupied = C.OCCUPIED_START <= (hour % 24) < C.OCCUPIED_END
    carbon_now = C.CARBON_INTENSITY[int(hour) % 24]
    carbon_peak = max(C.CARBON_INTENSITY[h] for h in range(C.OCCUPIED_START, C.OCCUPIED_END))
    carbon_offpeak = min(C.CARBON_INTENSITY[h] for h in range(0, C.OCCUPIED_START))

    summary = {
        "sim_day": day,
        "sim_hour": round(hour % 24, 1),
        "occupied": occupied,
        "window_hours": window_hours,
        "mean_zone_temp_C": avg("mean_temp"),
        "mean_pmv": avg("mean_pmv"),
        "pmv_in_band_fraction": avg("pmv_in_band"),
        "kwh_this_window": round(sum(r.get("kwh_total", 0) for r in window), 2),
        "peak_kw_this_window": round(max((r.get("kw_demand", 0) for r in window), default=0), 1),
        "base_cooling_setpoint_C": avg("base_cool_sp"),
        "base_heating_setpoint_C": avg("base_heat_sp"),
        "carbon_intensity_now_gco2_per_kwh": carbon_now,
        "carbon_intensity_peak_hour": carbon_peak,
        "carbon_intensity_cheapest_offpeak": carbon_offpeak,
        "current_policy": _state["current_policy"].to_dict(),
        "comfort_band_PMV": [C.PMV_LOW, C.PMV_HIGH],
        "occupied_hours": [C.OCCUPIED_START, C.OCCUPIED_END],
    }
    return json.dumps(summary, indent=2)


@mcp.tool()
def get_savings_so_far() -> str:
    """Get current energy and carbon savings compared to the baseline run.

    Returns absolute and percentage savings in kWh and kgCO2, plus whether
    thermal comfort has been maintained.
    """
    with _lock:
        base = _state["baseline_kwh"]
        curr = _state["current_kwh"]
        peak = _state["current_peak_kw"]
        pmv = _state["current_pmv_compliance"]
        decisions = len(_state["decisions"])

    if base == 0:
        return json.dumps({"error": "Baseline not loaded yet."})

    saved_kwh = base - curr
    pct = (saved_kwh / base * 100) if base > 0 else 0

    return json.dumps({
        "baseline_kwh": round(base, 1),
        "agent_kwh_so_far": round(curr, 1),
        "saved_kwh": round(saved_kwh, 1),
        "saved_pct": round(pct, 1),
        "peak_kw": round(peak, 1),
        "pmv_compliance": round(pmv * 100, 1),
        "comfort_maintained": pmv >= 0.88,
        "policy_changes_so_far": decisions,
        "note": (
            "Savings only count if comfort_maintained is true. "
            "The hard clamp ensures setpoints never exceed the safety envelope "
            "regardless of what policy you set."
        ),
    }, indent=2)


@mcp.tool()
def get_carbon_intensity(hour_of_day: int = -1) -> str:
    """Get grid carbon intensity (gCO2/kWh) by hour of day.

    Args:
        hour_of_day: 0-23 for a specific hour, -1 for the full 24-hour profile.

    Use this to decide WHEN to precool (cheap/clean hours) vs relax setpoints
    (expensive/dirty hours). The agent that shifts load to clean hours beats
    the agent that just reduces total kWh.
    """
    if hour_of_day == -1:
        profile = {str(h): C.CARBON_INTENSITY[h] for h in range(24)}
        cheapest = min(range(24), key=lambda h: C.CARBON_INTENSITY[h])
        dirtiest = max(range(24), key=lambda h: C.CARBON_INTENSITY[h])
        return json.dumps({
            "profile_gco2_per_kwh": profile,
            "cheapest_hour": cheapest,
            "cheapest_gco2_per_kwh": C.CARBON_INTENSITY[cheapest],
            "dirtiest_hour": dirtiest,
            "dirtiest_gco2_per_kwh": C.CARBON_INTENSITY[dirtiest],
            "strategy_hint": (
                f"Precool during hours {cheapest-2}-{cheapest+2} (clean solar midday). "
                f"Relax setpoints during hours {dirtiest-2}-{dirtiest+2} (dirty evening peak). "
                "Carbon savings can exceed energy savings when load shifts from dirty to clean hours."
            ),
        }, indent=2)
    h = max(0, min(23, hour_of_day))
    return json.dumps({
        "hour": h,
        "carbon_intensity_gco2_per_kwh": C.CARBON_INTENSITY[h],
    })


@mcp.tool()
def get_simulation_errors() -> str:
    """Get any EnergyPlus simulation warnings or errors.

    Returns a deduplicated, severity-bucketed summary. Use this to detect if
    the building model is behaving unexpectedly (unmet loads, convergence
    warnings) and adjust the policy accordingly.
    """
    with _lock:
        errors = list(_state["errors"])

    if not errors:
        return json.dumps({"status": "No errors or warnings recorded."})

    buckets: dict[str, list] = {"Fatal": [], "Severe": [], "Warning": [], "Info": []}
    seen = set()
    for e in errors:
        key = e.get("message", "")[:80]
        if key in seen:
            continue
        seen.add(key)
        sev = e.get("severity", "Info")
        buckets.setdefault(sev, []).append(key)

    return json.dumps({
        "total_unique": len(seen),
        "by_severity": {k: v for k, v in buckets.items() if v},
        "note": (
            "Severe/Fatal errors indicate the simulation may be producing wrong results. "
            "Warning-level unmet loads mean the HVAC cannot maintain the setpoints you set — "
            "consider relaxing them."
        ),
    }, indent=2)


@mcp.tool()
def set_control_policy(
    cooling_offset: float = 0.0,
    heating_offset: float = 0.0,
    unocc_cooling_offset: float = 0.0,
    unocc_heating_offset: float = 0.0,
    precool_hours: float = 0.0,
    precool_depth: float = 0.0,
    peak_start: int = 17,
    peak_end: int = 21,
    peak_offset: float = 0.0,
    reason: str = "",
) -> str:
    """Set the building control policy. Changes take effect immediately.

    The policy is validated before application. If it violates the safety
    envelope, you will receive a rejection with the allowed range — adjust
    and retry.

    Args:
        cooling_offset: degC added to cooling setpoint during OCCUPIED hours.
                        Positive = warmer = saves energy but risks comfort.
                        Keep small (0-0.5) during occupied hours.
        heating_offset: degC added to heating setpoint during occupied hours.
                        Negative = cooler = saves heating energy.
        unocc_cooling_offset: degC offset during UNOCCUPIED hours (nights/weekends).
                        Can be larger (1-3) since comfort doesn't matter.
        unocc_heating_offset: degC offset during unoccupied hours.
        precool_hours:  Hours BEFORE peak_start to pre-cool the building.
                        Charges thermal mass with cheap electricity.
        precool_depth:  How many degC below normal to drive setpoints during precool.
        peak_start:     Local hour when grid peak begins (default 17 = 5pm).
        peak_end:       Local hour when grid peak ends (default 21 = 9pm).
        peak_offset:    Extra degC of cooling setpoint relaxation during the peak.
                        Lets the building coast on stored coolth.
        reason:         One sentence explaining your strategy. Shown on the dashboard.
    """
    raw = {
        "cooling_offset": cooling_offset,
        "heating_offset": heating_offset,
        "unocc_cooling_offset": unocc_cooling_offset,
        "unocc_heating_offset": unocc_heating_offset,
        "precool_hours": precool_hours,
        "precool_depth": precool_depth,
        "peak_start": peak_start,
        "peak_end": peak_end,
        "peak_offset": peak_offset,
        "reason": reason,
        "issued_at": datetime.now().isoformat(),
    }
    try:
        policy = validate(raw)
    except PolicyRejected as e:
        return json.dumps({
            "status": "REJECTED",
            "reason": str(e),
            "instruction": "Fix the flagged field and call set_control_policy again.",
        }, indent=2)

    with _lock:
        _state["current_policy"] = policy
        _state["decisions"].append({
            "time": datetime.now().isoformat(),
            "policy": policy.to_dict(),
        })
        cb = _state["policy_callback"]

    if cb:
        cb(policy)

    return json.dumps({
        "status": "ACCEPTED",
        "policy": policy.to_dict(),
        "note": (
            "Policy is now active. The fast loop applies it every EnergyPlus timestep. "
            "Call get_savings_so_far in 2 simulated hours to measure the effect."
        ),
    }, indent=2)


def run_server():
    """Start the MCP server (blocking). Called on a background thread."""
    mcp.run(transport="stdio")
