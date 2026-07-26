"""The LLM agent — the brain of Eco-Loop.

Runs on a background thread, completely decoupled from the EnergyPlus simulation.
Every AGENT_INTERVAL simulated hours it:
  1. Calls get_telemetry() to read current building state
  2. Calls get_savings_so_far() to check progress
  3. Calls get_carbon_intensity() to see the grid
  4. Reasons about what policy to set
  5. Calls set_control_policy() with its decision
  6. If rejected: reads the error, corrects, retries once (self-correction loop)

The agent never blocks the simulation. If inference takes longer than the interval,
the last good policy keeps running. If the model returns garbage, the validator
rejects it and the last good policy keeps running. The simulation cannot stall.

Model: Qwen2.5-7B-Instruct via Ollama (best small-model tool calling as of 2025).
Fallback: any model Ollama has available.
"""

import json
import threading
import time
from typing import Any

from . import config as C
from .mcp_server import (
    get_telemetry,
    get_savings_so_far,
    get_carbon_intensity,
    set_control_policy,
    _state,
    _lock,
)

# How often the agent wakes up, in simulated hours.
# At 4 timesteps/hour a 2-hour interval = 8 timesteps between LLM calls.
# Adjust down if you want more frequent decisions for the demo video.
AGENT_INTERVAL_SIM_HOURS = 2.0

# Preferred models in order. First one Ollama has installed wins.
PREFERRED_MODELS = [
    "qwen2.5:7b-instruct",
    "qwen2.5:3b-instruct",
    "llama3.1:8b",
    "llama3:8b",
    "mistral:7b",
]

SYSTEM_PROMPT = """You are an expert building energy management AI controlling a real
office building through EnergyPlus simulation. Your goal is to minimize energy
consumption and carbon emissions while keeping occupant thermal comfort within
the ASHRAE 55 standard (PMV between -0.5 and +0.5 during occupied hours).

You have five tools available:
- get_telemetry: read current building sensor data
- get_savings_so_far: check energy and comfort performance vs baseline
- get_carbon_intensity: see the grid carbon profile by hour
- set_control_policy: update the building control strategy
- get_simulation_errors: check for any building model warnings

STRATEGY PRINCIPLES:
1. Occupied hours (07:00-19:00): keep PMV in [-0.5, +0.5]. Cooling offset max 0.5 degC.
2. Unoccupied hours: be aggressive. 1-3 degC setback is fine, nobody is there.
3. Precool before the dirty evening peak (17:00-21:00) using cheap clean midday electricity.
4. Carbon savings can exceed energy savings when you shift load from dirty to clean hours.
5. Watch the fan energy penalty — deep night setback causes morning pull-down spikes.
   Gradual recovery (small offsets) avoids this.
6. If set_control_policy is REJECTED, read the error, fix the value, retry immediately.

Always call get_telemetry first, then reason, then act. Explain your reasoning in
the `reason` field of set_control_policy — this appears on the dashboard.
"""


def _pick_model() -> str | None:
    """Return the first preferred model Ollama has available."""
    try:
        import ollama
        available = {m.model.split(":")[0] for m in ollama.list().models}
        for m in PREFERRED_MODELS:
            if m.split(":")[0] in available:
                return m
        # Fall back to whatever is installed
        models = ollama.list().models
        if models:
            return models[0].model
    except Exception:
        pass
    return None


def _run_agent_turn(model: str, sim_hour: float, sim_day: int, verbose: bool):
    """One agent decision cycle. Returns the policy dict or None on failure."""
    import ollama

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Initial prompt — what the agent sees at the start of its turn
    messages.append({"role": "user", "content": (
        f"It is simulated day {sim_day}, hour {sim_hour % 24:.1f}. "
        "Please check the building state, review the carbon profile, assess our "
        "savings so far, then set an optimal control policy. "
        "Explain your reasoning clearly in the reason field."
    )})

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_telemetry",
                "description": "Get recent building sensor readings summary",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "window_hours": {
                            "type": "number",
                            "description": "Hours of history to summarise (default 2)"
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_savings_so_far",
                "description": "Get energy and carbon savings vs baseline",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_carbon_intensity",
                "description": "Get grid carbon intensity by hour of day",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "hour_of_day": {
                            "type": "integer",
                            "description": "0-23 for a specific hour, -1 for full profile"
                        }
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_control_policy",
                "description": "Set the building control policy",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cooling_offset": {"type": "number"},
                        "heating_offset": {"type": "number"},
                        "unocc_cooling_offset": {"type": "number"},
                        "unocc_heating_offset": {"type": "number"},
                        "precool_hours": {"type": "number"},
                        "precool_depth": {"type": "number"},
                        "peak_start": {"type": "integer"},
                        "peak_end": {"type": "integer"},
                        "peak_offset": {"type": "number"},
                        "reason": {"type": "string"},
                    },
                    "required": ["reason"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_simulation_errors",
                "description": "Check for EnergyPlus warnings or errors",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]

    # Tool dispatch map
    tool_map = {
        "get_telemetry": lambda args: get_telemetry(**args),
        "get_savings_so_far": lambda args: get_savings_so_far(),
        "get_carbon_intensity": lambda args: get_carbon_intensity(**args),
        "set_control_policy": lambda args: set_control_policy(**args),
        "get_simulation_errors": lambda args: get_simulation_errors(),
    }

    MAX_TURNS = 8  # prevent infinite loops
    policy_set = False

    for turn in range(MAX_TURNS):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                tools=tools,
                options={"temperature": 0.2},  # low temp = more deterministic policy
            )
        except Exception as e:
            if verbose:
                print(f"  [agent] ollama error: {e}")
            return None

        msg = response.message
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in (msg.tool_calls or [])]})

        if not msg.tool_calls:
            # Model finished without calling set_control_policy
            if verbose:
                print(f"  [agent] finished after {turn+1} turns, policy_set={policy_set}")
            break

        for tc in msg.tool_calls:
            name = tc.function.name
            args = tc.function.arguments or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            if verbose:
                print(f"  [agent] -> {name}({list(args.keys())})")

            fn = tool_map.get(name)
            result = fn(args) if fn else json.dumps({"error": f"Unknown tool: {name}"})

            if verbose and name == "set_control_policy":
                r = json.loads(result)
                print(f"  [agent] policy {r['status']}: {args.get('reason','')[:80]}")
                if r["status"] == "ACCEPTED":
                    policy_set = True

            messages.append({
                "role": "tool",
                "content": result,
                "name": name,
            })

    return policy_set


class EcoAgent:
    """Background agent thread. Instantiate, call start(), then run the simulation."""

    def __init__(self, baseline_kwh: float, verbose: bool = True):
        self.baseline_kwh = baseline_kwh
        self.verbose = verbose
        self.model = None
        self._thread = None
        self._stop = threading.Event()
        self._last_sim_hour = -999.0

    def start(self):
        """Find the model and launch the background thread."""
        self.model = _pick_model()
        if self.model is None:
            print("  [agent] WARNING: no Ollama model found. "
                  "Run `ollama pull qwen2.5:7b-instruct` then restart.")
            return

        print(f"  [agent] using model: {self.model}")
        # Push the baseline so get_savings_so_far has a denominator
        with _lock:
            _state["baseline_kwh"] = self.baseline_kwh

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def notify_timestep(self, sim_hour: float, sim_day: int,
                        telemetry_row: dict, kwh_total: float,
                        peak_kw: float, pmv_compliance: float):
        """Called by the runner every timestep to push fresh state in."""
        with _lock:
            _state["sim_hour"] = sim_hour
            _state["sim_day"] = sim_day
            _state["current_kwh"] = kwh_total
            _state["current_peak_kw"] = peak_kw
            _state["current_pmv_compliance"] = pmv_compliance
            _state["telemetry"].append(telemetry_row)
            # Keep a rolling 24-hour window (96 timesteps at 4/hr)
            if len(_state["telemetry"]) > 96:
                _state["telemetry"] = _state["telemetry"][-96:]

    def _loop(self):
        """Agent loop: wake every AGENT_INTERVAL sim hours, run one decision cycle."""
        while not self._stop.is_set():
            with _lock:
                sim_hour = _state["sim_hour"]
                sim_day = _state["sim_day"]

            # Wait until enough simulated time has passed
            if sim_hour - self._last_sim_hour < AGENT_INTERVAL_SIM_HOURS:
                time.sleep(0.5)
                continue

            self._last_sim_hour = sim_hour
            if self.verbose:
                print(f"\n  [agent] waking: day {sim_day} hour {sim_hour % 24:.1f}")

            try:
                _run_agent_turn(self.model, sim_hour, sim_day, self.verbose)
            except Exception as e:
                if self.verbose:
                    print(f"  [agent] error in turn: {e}")
                # Last good policy keeps running — no crash, no stall
