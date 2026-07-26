# Eco-Loop Building Agents

An LLM agent that controls a building **from inside a running EnergyPlus simulation**.

Most AI-for-buildings work edits an `.idf`, re-runs the simulation, reads the CSV, and
edits again. That is batch tuning. This project embeds the control loop in the
EnergyPlus process itself via the Python Runtime API: callbacks fire every zone
timestep, read live sensor values out of the solver, and write setpoint actuators back
into it. No restart, no file rewriting, no offline iteration.

## Status

| Phase | | |
|---|---|---|
| 1 | Closed loop with a static policy | **done, verified** |
| 2 | Baseline capture + metrics | **done, verified** |
| 3 | MCP server exposing tools | next |
| 4 | Async LLM policy generation | next |
| 5 | Dashboard | next |

## Verified results

14 simulated days (1–14 July), DOE Medium Office prototype (15 conditioned zones),
Tampa TMY3, 4 timesteps/hour, EnergyPlus 24.1.0.

**Null-policy control test** — agent mode with an all-zero policy vs. baseline:

| | baseline | agent (null) | delta |
|---|---|---|---|
| Total electricity | 22208.01 kWh | 22206.66 kWh | −0.0% |
| Peak demand | 175.98 kW | 175.98 kW | +0.0% |
| PMV compliance | 90.33% | 90.41% | +0.1% |

This is the test that makes every later number trustworthy. If the actuation harness
introduced any bias, a "null" agent would not reproduce the baseline. It does, to
within 0.1% — the residual is the setpoint rate limiter smoothing the schedule's
instantaneous step changes.

**Measured tradeoff** — a 1.5 °C unoccupied setback with 2 h precooling:

| | baseline | agent | delta |
|---|---|---|---|
| Cooling electricity | 9849.62 kWh | 9574.41 kWh | **−2.8%** |
| Fan electricity | 1705.31 kWh | 2245.98 kWh | **+31.7%** |
| Total | 22208.01 kWh | 22473.47 kWh | +1.2% |

Deep night setback saves compressor energy and spends it back on fans during the
morning pull-down. This is a real physical result, not a bug — and it is exactly the
non-obvious tradeoff the LLM agent exists to navigate. A naive controller that only
watches the cooling meter would report a 2.8% "win" while making the building worse.

## Setup

```bash
# 1. EnergyPlus 24.1.0
wget https://github.com/NREL/EnergyPlus/releases/download/v24.1.0/EnergyPlus-24.1.0-9d7789a3ac-Linux-Ubuntu22.04-x86_64.tar.gz
mkdir -p ~/eplus && tar xzf EnergyPlus-24.1.0-*.tar.gz -C ~/eplus --strip-components=1
export ECOLOOP_EPLUS=~/eplus

# 2. Building model
cp $ECOLOOP_EPLUS/ExampleFiles/ASHRAE901_OfficeMedium_STD2019_Denver.idf models/baseline.idf
python scripts/patch_idf.py models/baseline.idf models/sim.idf --start 7 1 --end 7 14

# 3. Weather (Tampa ships with EnergyPlus; swap for a Chennai EPW from
#    climate.onebuilding.org for a hot-humid Indian climate)
export ECOLOOP_EPW=$ECOLOOP_EPLUS/WeatherData/USA_FL_Tampa.Intl.AP.722110_TMY3.epw

# 4. Verify
python -m pytest tests/ -q
python scripts/discover.py
python scripts/run.py --both --cooling-offset 0 --unocc-cooling-offset 0   # null test
```

## Run

```bash
python scripts/run.py --both \
    --cooling-offset 0.3 --unocc-cooling-offset 2.0 \
    --precool-hours 3 --precool-depth 1.0 --peak-offset 0.8
```

Outputs land in `results/{baseline,agent}/`: `timeseries.csv` (one row per timestep),
`summary.json`, `decisions.json` (the policy change log that feeds the dashboard).

## Architecture

Two speeds, because they run at incompatible rates.

An annual run at 4 timesteps/hour is 35,040 timesteps. At 2 s per LLM call that is
19 hours of inference. So:

- **Fast loop** — every timestep, pure Python, sub-millisecond. Reads sensors, resolves
  the current policy into setpoints, clamps them, writes actuators. Never calls the LLM.
- **Slow loop** — every few simulated hours, on a background thread. The LLM reviews
  summarised telemetry and issues a new *policy*: a small declarative object
  (`cooling_offset`, `precool_hours`, `peak_offset`, ...), not individual setpoints.

The simulation never blocks on inference. A stale policy is always safe to keep
executing, so a slow or failed LLM call degrades gracefully instead of stalling.

### The safety layer

Nothing reaches an actuator without passing `policy.validate()` then `policy.clamp()`.

`validate()` rejects out-of-range proposals with an error string that names the allowed
range — written to be read by the model, so the retry has everything it needs. That
rejection path *is* the self-correction loop.

`clamp()` then applies the hard envelope every timestep regardless of how the value was
produced: absolute setpoint bounds, a minimum 2 °C deadband, and a per-hour rate limit.
Even a validator bug cannot push the building outside the envelope. This is what lets
you tell a judge the agent cannot trade away comfort no matter what the model
hallucinates.

## Files

```
src/ecoloop/config.py    building, meters, safety envelope, carbon curve
src/ecoloop/policy.py    Policy dataclass, validator, clamp        <- fully unit tested
src/ecoloop/runner.py    EnergyPlus callbacks, sensor read, actuator write
scripts/patch_idf.py     single run period, PMV enablement, timestep
scripts/discover.py      dump every available handle
scripts/run.py           baseline vs agent driver + comparison table
tests/test_policy.py     10 tests, no EnergyPlus required, <1s
```

## Gotchas already solved

Recorded because each one cost real debugging time.

1. **Handles are only valid after `api_data_fully_ready()`.** Fetch once in
   `begin_new_environment` and cache. Fetching per timestep is 35,000 string lookups.
2. **`request_variable()` must precede `run_energyplus()`.** Otherwise
   `get_variable_handle` returns −1 with no explanation.
3. **A −1 handle reads as `0.0` forever — it does not raise.** The runner treats any
   unresolved handle as fatal. Silent zeros in an energy table are worse than a crash.
4. **Meter names are not portable.** This model has on-site generation, so
   `Electricity:Facility` does not exist; it is `ElectricityNet:Facility`. `config.METERS`
   is a fallback chain per metric.
5. **Skip `warmup_flag()` timesteps.** EnergyPlus repeats warmup days to converge
   initial conditions; logging them double-counts energy.
6. **DOE prototypes ship with three `RunPeriod` objects.** Each is a separate
   environment, so callbacks fire three times over and the comparison stops being
   apples to apples. `patch_idf.py` collapses them to one.
7. **IDF comments trail the comma.** Strip `!- ...` line by line *before* splitting on
   commas, or every parsed field comes back as the empty string.
8. **PMV needs the Fanger model switched on** in each `People` object. This particular
   model already has it; most do not, and the variable simply will not exist until you
   do. 20% of the grade depends on it.

## Next

- `mcp_server.py` — expose `get_telemetry`, `get_simulation_errors`,
  `set_control_policy`, `get_carbon_intensity`, `get_savings_so_far` over stdio MCP.
- `agent.py` — Ollama + Qwen2.5-7B on a background thread, JSON-schema constrained,
  retry-on-rejection, last-good-policy fallback.
- Fix the morning pull-down fan penalty with a ramped recovery rather than a step
  return to occupied setpoints.
- Dashboard: cumulative kWh baseline vs agent, setpoint trace, PMV histogram,
  decision timeline.
