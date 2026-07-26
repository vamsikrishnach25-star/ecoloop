"""The closed loop itself.

This is the file that makes the project what it claims to be. The LLM agent runs
*inside* the EnergyPlus process via the Python Runtime API: callbacks fire at
every zone timestep, read live sensor values out of the running simulation, and
write setpoint actuators back into it. No file rewriting, no restarting, no
batch iteration.

Callback order in one EnergyPlus zone timestep:

    begin_new_environment                <- handles become valid here, once
    begin_zone_timestep_after_init_heat_balance
                                         <- WE ACT HERE: read sensors, write setpoints
    ... EnergyPlus solves the zone/system/plant ...
    end_zone_timestep_after_zone_reporting
                                         <- WE MEASURE HERE: meters have settled

Three details that cost people hours:

  1. Handles are integers valid only after `api_data_fully_ready()` returns true.
     Fetch them in `begin_new_environment` and cache them. Fetching every
     timestep is a string lookup 35,000 times over and is measurably slow.
  2. `warmup_flag()` is true during warmup days, which EnergyPlus runs repeatedly
     to converge initial conditions. Controlling during warmup corrupts nothing
     but logging during warmup produces duplicate rows and a nonsense energy total.
  3. `request_variable()` must be called BEFORE `run_energyplus()`. Any variable
     not in the IDF's Output:Variable list simply will not exist otherwise, and
     `get_variable_handle` returns -1 with no explanation.
"""

import csv
import sys
from pathlib import Path

from . import config as C
from .policy import Policy, target_setpoints, clamp

# The DOE prototype shares one setpoint schedule pair across all zones. Reading
# these gives us the un-actuated scheduled setpoint even while we are overriding
# each zone's thermostat -- that is what makes the offsets in Policy meaningful
# and keeps the agent's night setback aligned with the building's own schedule.
CLG_SCHEDULE = "CLGSETP_SCH_YES_OPTIMUM"
HTG_SCHEDULE = "HTGSETP_SCH_YES_OPTIMUM"

CSV_HEADER = (
    ["step", "month", "day", "hour",
     "kwh_total", "kwh_cooling", "kwh_heating", "kwh_fans", "kwh_lights",
     "kw_demand", "base_cool_sp", "base_heat_sp",
     "mean_temp", "mean_pmv", "pmv_in_band", "occupied", "policy_reason"]
)


class EcoLoopRunner:
    """Runs one EnergyPlus simulation with an optional live control policy.

    mode='baseline'  -- no actuation at all. Produces the denominator.
    mode='agent'     -- applies whatever policy `policy_provider()` returns.

    `policy_provider` is a zero-argument callable returning a Policy. In Phase 1
    that is a lambda returning a fixed Policy. In Phase 4 it is the LLM agent's
    `current_policy` property, updated asynchronously on a background thread.
    The runner does not know or care which -- that is the seam that keeps the
    simulation from ever blocking on inference.
    """

    def __init__(self, idf, epw, out_dir, mode="baseline",
                 policy_provider=None, eplus_dir=None, verbose=False):
        self.idf = str(idf)
        self.epw = str(epw)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.policy_provider = policy_provider or (lambda: Policy())
        self.verbose = verbose

        eplus_dir = str(eplus_dir or C.EPLUS_DIR)
        if not Path(eplus_dir).is_dir():
            raise FileNotFoundError(
                f"EnergyPlus not found at {eplus_dir}.\n"
                f"Set ECOLOOP_EPLUS to your install directory -- the folder that "
                f"contains the 'pyenergyplus' package."
            )
        if eplus_dir not in sys.path:
            sys.path.insert(0, eplus_dir)

        # Windows: Python 3.8+ stopped searching PATH for dependent DLLs. The
        # absolute-path load of EnergyPlusAPI.dll succeeds, then fails to resolve
        # the DLLs sitting next to it, and you get an opaque OSError with error
        # code 126. Registering the directory fixes it. No-op elsewhere.
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(eplus_dir)

        self.h = {}                # cached handles
        self.meter_names = {}      # which candidate name actually resolved
        self.ready = False
        self.step = 0
        self.prev_sp = {}          # zone -> (cool, heat), for rate limiting
        self.rows = []
        self.totals = {k: 0.0 for k in C.METERS}
        self.peak_kw = 0.0
        self.pmv_occupied = []     # every occupied-hour PMV reading
        self.decisions = []        # policy changes, for the dashboard timeline
        self._last_reason = None

    # -- callbacks ------------------------------------------------------------

    def _on_new_environment(self, state):
        """Cache every handle exactly once, the moment the data is available."""
        ex = self.api.exchange
        if not ex.api_data_fully_ready(state):
            return

        self.h = {
            "temp": {z: ex.get_variable_handle(state, "Zone Mean Air Temperature", z)
                     for z in C.ZONES},
            "pmv": {z: ex.get_variable_handle(
                        state, "Zone Thermal Comfort Fanger Model PMV", z)
                    for z in C.ZONES},
            "cool_sp": {z: ex.get_actuator_handle(
                            state, "Zone Temperature Control", "Cooling Setpoint", z)
                        for z in C.ZONES},
            "heat_sp": {z: ex.get_actuator_handle(
                            state, "Zone Temperature Control", "Heating Setpoint", z)
                        for z in C.ZONES},
            "meters": {},
            "base_cool": ex.get_variable_handle(state, "Schedule Value", CLG_SCHEDULE),
            "base_heat": ex.get_variable_handle(state, "Schedule Value", HTG_SCHEDULE),
        }

        for key, candidates in C.METERS.items():
            for name in candidates:
                handle = ex.get_meter_handle(state, name)
                if handle >= 0:
                    self.h["meters"][key] = handle
                    self.meter_names[key] = name
                    break
            else:
                self.h["meters"][key] = -1
                self.meter_names[key] = None

        bad = []
        for group in ("temp", "pmv", "cool_sp", "heat_sp"):
            bad += [f"{group}:{z}" for z, v in self.h[group].items() if v < 0]
        bad += [f"meter:{k}" for k, v in self.h["meters"].items() if v < 0]
        for k in ("base_cool", "base_heat"):
            if self.h[k] < 0:
                bad.append(f"schedule:{k}")

        if bad:
            # Loud, and fatal. A -1 handle reads as 0.0 forever, which does not
            # crash -- it just produces a confident, completely wrong answer.
            # That is the worst possible failure mode for a results table.
            raise RuntimeError(
                f"{len(bad)} handle(s) failed to resolve: {bad}\n"
                f"Run `python scripts/discover.py` and check the names in config.py."
            )

        self.ready = True

    def _on_timestep(self, state):
        """Read sensors, resolve the policy, inject setpoints. The control step."""
        ex = self.api.exchange
        if not self.ready or ex.warmup_flag(state):
            return

        hour = ex.current_time(state)          # decimal hour, 0-24
        base_cool = ex.get_variable_value(state, self.h["base_cool"])
        base_heat = ex.get_variable_value(state, self.h["base_heat"])
        self._base = (base_cool, base_heat)

        if self.mode != "agent":
            self._policy = Policy(reason="baseline: scheduled setpoints, no override")
            return

        policy = self.policy_provider()
        self._policy = policy

        if policy.reason != self._last_reason:
            self.decisions.append({
                "step": self.step, "hour": round(hour, 2),
                "reason": policy.reason, "policy": policy.to_dict(),
            })
            self._last_reason = policy.reason

        dt = ex.zone_time_step(state)          # length of this timestep, in hours
        for z in C.ZONES:
            cool, heat = target_setpoints(policy, z, hour, base_cool, base_heat)
            pc, ph = self.prev_sp.get(z, (None, None))
            cool, heat = clamp(cool, heat, pc, ph, dt_hours=dt)
            ex.set_actuator_value(state, self.h["cool_sp"][z], cool)
            ex.set_actuator_value(state, self.h["heat_sp"][z], heat)
            self.prev_sp[z] = (cool, heat)

    def _on_report(self, state):
        """Meters have settled -- measure and log."""
        ex = self.api.exchange
        if not self.ready or ex.warmup_flag(state):
            return

        joules = {k: ex.get_meter_value(state, h) for k, h in self.h["meters"].items()}
        kwh = {k: v * C.J_TO_KWH for k, v in joules.items()}
        for k, v in kwh.items():
            self.totals[k] += v

        # Demand over a 15-minute timestep: kWh in the interval / interval hours.
        kw = kwh["total"] / 0.25
        self.peak_kw = max(self.peak_kw, kw)

        temps, pmvs = [], []
        for z in C.ZONES:
            temps.append(ex.get_variable_value(state, self.h["temp"][z]))
            pmvs.append(ex.get_variable_value(state, self.h["pmv"][z]))

        hour = ex.current_time(state)
        occupied = C.OCCUPIED_START <= hour < C.OCCUPIED_END
        mean_pmv = sum(pmvs) / len(pmvs)
        in_band = sum(1 for p in pmvs if C.PMV_LOW <= p <= C.PMV_HIGH) / len(pmvs)
        if occupied:
            self.pmv_occupied.extend(pmvs)

        self.rows.append([
            self.step, ex.month(state), ex.day_of_month(state), round(hour, 3),
            kwh["total"], kwh["cooling"], kwh["heating"], kwh["fans"], kwh["lights"],
            kw, round(self._base[0], 3), round(self._base[1], 3),
            round(sum(temps) / len(temps), 3), round(mean_pmv, 4),
            round(in_band, 4), int(occupied), self._policy.reason,
        ])
        self.step += 1

        if self.verbose and self.step % 96 == 0:
            print(f"    day {self.step // 96:>2}  "
                  f"{self.totals['total']:8.1f} kWh  "
                  f"peak {self.peak_kw:5.1f} kW  "
                  f"PMV {mean_pmv:+.2f}")

    # -- run ------------------------------------------------------------------

    def run(self):
        from pyenergyplus.api import EnergyPlusAPI

        self.api = EnergyPlusAPI()
        state = self.api.state_manager.new_state()
        rt, ex = self.api.runtime, self.api.exchange

        # MUST precede run_energyplus, or these variables never exist.
        for z in C.ZONES:
            ex.request_variable(state, "Zone Mean Air Temperature", z)
            ex.request_variable(state, "Zone Thermal Comfort Fanger Model PMV", z)
        ex.request_variable(state, "Schedule Value", CLG_SCHEDULE)
        ex.request_variable(state, "Schedule Value", HTG_SCHEDULE)

        rt.callback_begin_new_environment(state, self._on_new_environment)
        rt.callback_begin_zone_timestep_after_init_heat_balance(state, self._on_timestep)
        rt.callback_end_zone_timestep_after_zone_reporting(state, self._on_report)
        rt.set_console_output_status(state, False)

        code = rt.run_energyplus(
            state, ["-d", str(self.out_dir), "-w", self.epw, self.idf])
        self.api.state_manager.delete_state(state)

        if code != 0:
            raise RuntimeError(
                f"EnergyPlus exited {code}. See {self.out_dir / 'eplusout.err'}")
        self._write_csv()
        return self.summary()

    def _write_csv(self):
        path = self.out_dir / "timeseries.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(CSV_HEADER)
            w.writerows(self.rows)

    def summary(self):
        n = len(self.pmv_occupied) or 1
        comfy = sum(1 for p in self.pmv_occupied if C.PMV_LOW <= p <= C.PMV_HIGH)
        carbon = sum(
            r[4] * C.CARBON_INTENSITY[int(r[3]) % 24] for r in self.rows) / 1000.0
        return {
            "mode": self.mode,
            "timesteps": len(self.rows),
            "kwh_total": self.totals["total"],
            "kwh_cooling": self.totals["cooling"],
            "kwh_fans": self.totals["fans"],
            "peak_kw": self.peak_kw,
            "pmv_compliance": comfy / n,
            "kgco2": carbon,
            "decisions": len(self.decisions),
        }
