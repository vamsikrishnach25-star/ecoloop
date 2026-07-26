"""Run the full Eco-Loop: EnergyPlus + live LLM agent.

    python scripts/run_agent.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecoloop import config as C
from ecoloop.agent import EcoAgent
from ecoloop.mcp_server import update_state, get_current_policy
from ecoloop.policy import Policy
from ecoloop.runner import EcoLoopRunner


def run_baseline():
    print("\n" + "="*60)
    print("  PHASE 1: Baseline (no agent)")
    print("="*60)
    runner = EcoLoopRunner(
        idf=C.IDF, epw=C.EPW,
        out_dir=C.RESULTS / "baseline",
        mode="baseline", verbose=True,
    )
    summary = runner.run()
    print(f"\n  Baseline: {summary['kwh_total']:.1f} kWh  "
          f"peak {summary['peak_kw']:.1f} kW  "
          f"PMV {summary['pmv_compliance']*100:.1f}%")
    (C.RESULTS / "baseline" / "summary.json").write_text(
        json.dumps(summary, indent=2))
    return summary


def run_with_agent(baseline_kwh: float):
    print("\n" + "="*60)
    print("  PHASE 2: Agent control (LLM in the loop)")
    print("="*60)

    agent = EcoAgent(baseline_kwh=baseline_kwh, verbose=True)

    # IMPORTANT: Pre-seed with a known-good policy so the simulation
    # gets real agent control even before the first LLM response arrives.
    # The LLM then refines this every 2 simulated hours.
    from ecoloop.mcp_server import _state, _lock
    with _lock:
        _state["baseline_kwh"] = baseline_kwh
        _state["current_policy"] = Policy(
            cooling_offset=0.0,
            unocc_cooling_offset=0.8,
            unocc_heating_offset=-0.5,
            precool_hours=2.0,
            precool_depth=0.6,
            peak_start=17,
            peak_end=21,
            peak_offset=0.4,
            reason="initial policy: unoccupied setback + precool before evening peak",
        )

    agent.start()

    out_dir = C.RESULTS / "agent"
    runner = EcoLoopRunner(
        idf=C.IDF, epw=C.EPW,
        out_dir=out_dir,
        mode="agent",
        policy_provider=get_current_policy,
        verbose=True,
    )

    # Hook into runner to feed agent every timestep
    original_report = runner._on_report
    def patched_report(state):
        original_report(state)
        if runner.rows:
            row = runner.rows[-1]
            agent.notify_timestep(
                sim_hour=row[3],
                sim_day=row[2],
                telemetry_row={
                    "mean_temp": row[12],
                    "mean_pmv": row[13],
                    "pmv_in_band": row[14],
                    "kwh_total": row[4],
                    "kw_demand": row[9],
                    "base_cool_sp": row[10],
                    "base_heat_sp": row[11],
                },
                kwh_total=runner.totals["total"],
                peak_kw=runner.peak_kw,
                pmv_compliance=(
                    sum(1 for p in runner.pmv_occupied
                        if C.PMV_LOW <= p <= C.PMV_HIGH)
                    / max(len(runner.pmv_occupied), 1)
                ),
            )
    runner._on_report = patched_report

    summary = runner.run()
    agent.stop()

    summary["decisions"] = len(runner.decisions)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "decisions.json").write_text(
        json.dumps(runner.decisions, indent=2))
    return summary


def print_table(base, agent_s):
    def row(label, b, a, unit, better="down"):
        if b == 0:
            pct = "--"
        else:
            d = (a - b) / b * 100
            pct = f"{d:+.1f}%"
        good = (a < b) if better == "down" else (a > b)
        mark = "OK " if good else "!! "
        return f"  {mark}{label:<28}{b:>10.2f}{a:>12.2f}{unit:>6}  {pct:>8}"

    print("\n" + "="*72)
    print(f"  {'metric':<30}{'baseline':>10}{'agent':>12}{'':>6}{'change':>10}")
    print("-"*72)
    print(row("total electricity", base["kwh_total"], agent_s["kwh_total"], "kWh"))
    print(row("cooling electricity", base["kwh_cooling"], agent_s["kwh_cooling"], "kWh"))
    print(row("fan electricity", base["kwh_fans"], agent_s["kwh_fans"], "kWh"))
    print(row("peak demand", base["peak_kw"], agent_s["peak_kw"], "kW"))
    print(row("carbon", base["kgco2"], agent_s["kgco2"], "kgCO2"))
    print(row("PMV compliance %", base["pmv_compliance"]*100,
              agent_s["pmv_compliance"]*100, "%", better="up"))
    print(f"  {'LLM decisions':<30}{'0':>10}{agent_s['decisions']:>12}{'':>6}")
    print("="*72)

    comfort_ok = agent_s["pmv_compliance"] >= base["pmv_compliance"] - 0.02
    if comfort_ok:
        print("\n  ✓ Comfort maintained. Energy savings are valid.")
    else:
        print("\n  ✗ Comfort degraded vs baseline.")
    print()


def main():
    try:
        import ollama
        models = ollama.list().models
        if models:
            print(f"  Ollama ready. Models: {[m.model for m in models]}")
        else:
            print("  WARNING: No Ollama models. Run: ollama pull qwen2.5:1.5b-instruct")
    except Exception as e:
        print(f"  WARNING: Ollama not reachable ({e})")

    C.RESULTS.mkdir(parents=True, exist_ok=True)

    baseline_cache = C.RESULTS / "baseline" / "summary.json"
    if baseline_cache.exists():
        base = json.loads(baseline_cache.read_text())
        print(f"\n  Using cached baseline: {base['kwh_total']:.1f} kWh")
    else:
        base = run_baseline()

    agent_s = run_with_agent(base["kwh_total"])
    print_table(base, agent_s)


if __name__ == "__main__":
    main()
