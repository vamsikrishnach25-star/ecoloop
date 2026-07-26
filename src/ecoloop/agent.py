"""LLM agent — works reliably with small models (1.5b+).

Instead of complex tool-calling chains (which need 7b+ models), this agent:
1. Gathers telemetry directly from shared state
2. Builds a rich context prompt
3. Asks the LLM for a JSON policy decision
4. Validates and applies the policy
5. Retries with error feedback if rejected (self-correction loop)
"""

import json
import threading
import time

from . import config as C
from .mcp_server import (
    get_telemetry, get_savings_so_far, get_carbon_intensity,
    set_control_policy, get_current_policy, _state, _lock,
)

AGENT_INTERVAL_SIM_HOURS = 0.5

PREFERRED_MODELS = [
    "qwen2.5:1.5b-instruct",
    "qwen2.5:7b-instruct",
    "qwen2.5:3b-instruct",
    "llama3.1:8b",
    "llama3:8b",
    "mistral:7b",
]

POLICY_PROMPT = """You are an AI building energy manager. Based on the sensor data below,
output a JSON control policy to minimize energy and carbon while keeping PMV in [-0.5, +0.5]
during occupied hours (07:00-19:00).

CURRENT BUILDING STATE:
{telemetry}

SAVINGS SO FAR:
{savings}

CARBON NOW:
{carbon}

OUTPUT RULES:
- Return ONLY a valid JSON object, no text outside the JSON
- Use the "reason" field to explain your strategy in one sentence
- During occupied hours (07-19): cooling_offset max 0.5
- During unoccupied hours: unocc_cooling_offset can be 0.5-2.0
- If carbon > 700: use peak_offset to reduce peak load
- If carbon < 600: use precool to store cheap clean energy

EXAMPLE:
{{
  "cooling_offset": 0.2,
  "heating_offset": 0.0,
  "unocc_cooling_offset": 1.0,
  "unocc_heating_offset": -0.5,
  "precool_hours": 2.0,
  "precool_depth": 0.8,
  "peak_start": 17,
  "peak_end": 21,
  "peak_offset": 0.5,
  "reason": "Carbon high at peak, precooling with clean midday electricity"
}}

YOUR JSON POLICY:"""


def _pick_model():
    try:
        import ollama
        available = {m.model for m in ollama.list().models}
        for m in PREFERRED_MODELS:
            if m in available:
                return m
        for m in PREFERRED_MODELS:
            for a in available:
                if a.startswith(m.split(":")[0]):
                    return a
        if available:
            return list(available)[0]
    except Exception:
        pass
    return None


def _ask_llm(model, prompt):
    import ollama
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        text = response.message.content.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return None


def _run_agent_turn(model, sim_hour, sim_day, verbose):
    telemetry = get_telemetry(window_hours=2.0)
    savings = get_savings_so_far()
    carbon = get_carbon_intensity(hour_of_day=int(sim_hour) % 24)

    prompt = POLICY_PROMPT.format(
        telemetry=telemetry, savings=savings, carbon=carbon)

    if verbose:
        print(f"  [agent] asking {model} for policy...")

    raw = _ask_llm(model, prompt)
    if raw is None:
        if verbose:
            print("  [agent] no valid JSON returned, keeping current policy")
        return False

    if verbose:
        print(f"  [agent] proposed: {raw.get('reason','')[:80]}")

    result_str = set_control_policy(
        cooling_offset=float(raw.get("cooling_offset", 0)),
        heating_offset=float(raw.get("heating_offset", 0)),
        unocc_cooling_offset=float(raw.get("unocc_cooling_offset", 0)),
        unocc_heating_offset=float(raw.get("unocc_heating_offset", 0)),
        precool_hours=float(raw.get("precool_hours", 0)),
        precool_depth=float(raw.get("precool_depth", 0)),
        peak_start=int(raw.get("peak_start", 17)),
        peak_end=int(raw.get("peak_end", 21)),
        peak_offset=float(raw.get("peak_offset", 0)),
        reason=str(raw.get("reason", "LLM policy")),
    )
    result = json.loads(result_str)

    if result["status"] == "REJECTED":
        if verbose:
            print(f"  [agent] REJECTED: {result['reason']}")
        # Self-correction: retry with error context
        retry_prompt = (prompt +
            f"\n\nREJECTED: {result['reason']}\nFix and output corrected JSON:")
        raw2 = _ask_llm(model, retry_prompt)
        if raw2:
            r2 = set_control_policy(
                cooling_offset=float(raw2.get("cooling_offset", 0)),
                heating_offset=float(raw2.get("heating_offset", 0)),
                unocc_cooling_offset=float(raw2.get("unocc_cooling_offset", 0)),
                unocc_heating_offset=float(raw2.get("unocc_heating_offset", 0)),
                precool_hours=float(raw2.get("precool_hours", 0)),
                precool_depth=float(raw2.get("precool_depth", 0)),
                peak_start=int(raw2.get("peak_start", 17)),
                peak_end=int(raw2.get("peak_end", 21)),
                peak_offset=float(raw2.get("peak_offset", 0)),
                reason=str(raw2.get("reason", "corrected policy")),
            )
            r2d = json.loads(r2)
            if verbose:
                print(f"  [agent] retry {r2d['status']}")
            return r2d["status"] == "ACCEPTED"
        return False

    if verbose:
        print(f"  [agent] ACCEPTED ✓")
    return True


class EcoAgent:
    def __init__(self, baseline_kwh, verbose=True):
        self.baseline_kwh = baseline_kwh
        self.verbose = verbose
        self.model = None
        self._thread = None
        self._stop = threading.Event()
        self._last_sim_hour = -999.0

    def start(self):
        self.model = _pick_model()
        if self.model is None:
            print("  [agent] WARNING: no Ollama model found.")
            return
        print(f"  [agent] using model: {self.model}")
        with _lock:
            _state["baseline_kwh"] = self.baseline_kwh
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def notify_timestep(self, sim_hour, sim_day, telemetry_row,
                        kwh_total, peak_kw, pmv_compliance):
        with _lock:
            _state["sim_hour"] = sim_hour
            _state["sim_day"] = sim_day
            _state["current_kwh"] = kwh_total
            _state["current_peak_kw"] = peak_kw
            _state["current_pmv_compliance"] = pmv_compliance
            _state["telemetry"].append(telemetry_row)
            if len(_state["telemetry"]) > 96:
                _state["telemetry"] = _state["telemetry"][-96:]

    def _loop(self):
        while not self._stop.is_set():
            with _lock:
                sim_hour = _state["sim_hour"]
                sim_day = _state["sim_day"]
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
                    print(f"  [agent] error: {e}")
