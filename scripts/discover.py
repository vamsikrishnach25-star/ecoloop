"""Dump every actuator, variable and meter this IDF exposes to the API.

Run this FIRST whenever you change building model or EnergyPlus version. Handle
names are not portable between models -- `Electricity:Facility` does not exist in
a building with on-site generation, and a missing handle returns -1, which reads
as 0.0 forever instead of raising. Silent zeros in an energy table are the worst
bug this project can have.

    python scripts/discover.py                 # writes results/api_data.csv
    python scripts/discover.py --grep setpoint
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ecoloop import config as C

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grep", default=None, help="case-insensitive filter")
    a = p.parse_args()

    sys.path.insert(0, str(C.EPLUS_DIR))
    from pyenergyplus.api import EnergyPlusAPI

    api = EnergyPlusAPI(); state = api.state_manager.new_state(); done = {"v": False}
    out = C.RESULTS / "api_data.csv"; out.parent.mkdir(parents=True, exist_ok=True)

    def dump(s):
        if done["v"] or not api.exchange.api_data_fully_ready(s):
            return
        done["v"] = True
        out.write_bytes(api.exchange.list_available_api_data_csv(s))
        api.runtime.stop_simulation(s)

    api.runtime.callback_begin_zone_timestep_after_init_heat_balance(state, dump)
    api.runtime.set_console_output_status(state, False)
    api.runtime.run_energyplus(
        state, ["-d", str(C.RESULTS / "_discover"), "-w", str(C.EPW), str(C.IDF)])

    if not out.exists():
        print("no dump produced -- check results/_discover/eplusout.err", file=sys.stderr)
        return 1
    lines = out.read_text(errors="replace").splitlines()
    print(f"{len(lines)} entries -> {out}")
    if a.grep:
        for ln in lines:
            if a.grep.lower() in ln.lower():
                print(" ", ln)
    else:
        from collections import Counter
        for k, n in Counter(l.split(",")[0] for l in lines).most_common():
            print(f"  {n:>6}  {k}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
