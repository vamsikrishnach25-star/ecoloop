"""Run baseline and/or agent, and print the comparison table.

    python scripts/run.py --both
    python scripts/run.py --mode agent --cooling-offset 1.5 --precool-hours 3
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecoloop import config as C          # noqa: E402
from ecoloop.policy import Policy        # noqa: E402
from ecoloop.runner import EcoLoopRunner  # noqa: E402


def run_one(mode, policy, verbose):
    out = C.RESULTS / mode
    runner = EcoLoopRunner(
        idf=C.IDF, epw=C.EPW, out_dir=out, mode=mode,
        policy_provider=(lambda: policy), verbose=verbose,
    )
    print(f"\n>> {mode}")
    s = runner.run()
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    (out / "decisions.json").write_text(json.dumps(runner.decisions, indent=2))
    return s


def table(base, agent):
    def row(label, b, a, unit="", better="down"):
        if b == 0:
            pct = "--"
        else:
            d = (a - b) / b * 100
            pct = f"{d:+.1f}%"
        good = (a < b) if better == "down" else (a > b)
        mark = "OK " if good else "!! "
        return f"  {mark}{label:<26}{b:>11.2f}{a:>13.2f}{unit:>7}  {pct:>8}"

    print("\n" + "=" * 74)
    print(f"  {'metric':<28}{'baseline':>11}{'agent':>13}{'':>7}{'change':>10}")
    print("-" * 74)
    print(row("total electricity", base["kwh_total"], agent["kwh_total"], "kWh"))
    print(row("cooling electricity", base["kwh_cooling"], agent["kwh_cooling"], "kWh"))
    print(row("fan electricity", base["kwh_fans"], agent["kwh_fans"], "kWh"))
    print(row("peak demand", base["peak_kw"], agent["peak_kw"], "kW"))
    print(row("carbon", base["kgco2"], agent["kgco2"], "kgCO2"))
    print(row("PMV compliance (occ.)", base["pmv_compliance"] * 100,
              agent["pmv_compliance"] * 100, "%", better="up"))
    print("=" * 74)
    if agent["pmv_compliance"] < base["pmv_compliance"] - 0.005:
        print("  WARNING: comfort degraded vs baseline. Energy savings do not count.")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "agent"], default="agent")
    p.add_argument("--both", action="store_true")
    p.add_argument("--cooling-offset", type=float, default=1.0)
    p.add_argument("--heating-offset", type=float, default=-1.0)
    p.add_argument("--precool-hours", type=float, default=0.0)
    p.add_argument("--precool-depth", type=float, default=0.0)
    p.add_argument("--peak-offset", type=float, default=0.0)
    p.add_argument("--unocc-cooling-offset", type=float, default=0.0)
    p.add_argument("--unocc-heating-offset", type=float, default=0.0)
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args()

    policy = Policy(
        cooling_offset=a.cooling_offset,
        heating_offset=a.heating_offset,
        precool_hours=a.precool_hours,
        precool_depth=a.precool_depth,
        peak_offset=a.peak_offset,
        unocc_cooling_offset=a.unocc_cooling_offset,
        unocc_heating_offset=a.unocc_heating_offset,
        reason=("static phase-1 policy: "
                f"cool{a.cooling_offset:+g} heat{a.heating_offset:+g} "
                f"precool {a.precool_hours:g}h/-{a.precool_depth:g}C "
                f"peak+{a.peak_offset:g}C"),
    )

    if a.both:
        base = run_one("baseline", Policy(), a.verbose)
        ag = run_one("agent", policy, a.verbose)
        table(base, ag)
    else:
        s = run_one(a.mode, policy, a.verbose)
        print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
