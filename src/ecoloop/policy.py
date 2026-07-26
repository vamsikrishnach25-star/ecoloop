"""The control policy, and the validator that guards it.

The single most important design decision in this project: the LLM does not
write setpoints. It writes a POLICY -- a small, declarative object describing a
strategy. The fast loop then executes that policy every timestep in pure Python.

Why this matters:
  * An annual run at 4 timesteps/hour is 35,040 timesteps. At 2 s per LLM call
    that is 19 hours of inference. Policies let one call cover many timesteps.
  * A stale policy is still a safe policy, so the simulation never has to block
    waiting for the model.
  * A policy is small enough to validate exhaustively. A stream of raw setpoints
    is not.

Everything the model proposes passes through `validate()` before it can touch an
actuator. When validation fails the reason string goes back to the model as a
tool result -- that is the self-correction loop.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional

from . import config as C


class PolicyRejected(Exception):
    """Raised when a proposed policy violates the safety envelope."""


@dataclass
class Policy:
    """A declarative control strategy the fast loop can execute.

    cooling_offset : degC added to the scheduled cooling setpoint during
                     OCCUPIED hours. Positive = warmer = cheaper, but this is
                     the offset that costs comfort, so it must stay small.
    heating_offset : degC added to the scheduled heating setpoint, occupied hours.
    unocc_*_offset : the same, for UNOCCUPIED hours. These are close to free --
                     nobody is in the building to be uncomfortable -- so they can
                     be far more aggressive. Separating occupied from unoccupied
                     is what lets the agent cut energy without cutting comfort.
    precool_hours  : hours before `peak_start` to drive setpoints DOWN by
                     `precool_depth`, charging the building's thermal mass with
                     cheap clean electricity so the compressor can coast
                     through the dirty evening peak.
    peak_start/end : local hours bounding the grid peak.
    peak_offset    : extra degC of setpoint relaxation during the peak window.
    zones          : which zones this applies to. None means all conditioned zones.
    reason         : the model's own one-line justification. Not used for control,
                     but logged and shown on the dashboard -- this is what makes
                     the demo legible to a judge.
    """

    cooling_offset: float = 0.0
    heating_offset: float = 0.0
    unocc_cooling_offset: float = 0.0
    unocc_heating_offset: float = 0.0
    precool_hours: float = 0.0
    precool_depth: float = 0.0
    peak_start: int = 17
    peak_end: int = 21
    peak_offset: float = 0.0
    zones: Optional[list] = None
    reason: str = "baseline hold"
    issued_at: str = ""

    def to_dict(self):
        return asdict(self)


# Bounds on what the model is allowed to propose, before the per-timestep clamp.
# Deliberately tighter than the hard envelope in config so a bad policy is
# rejected loudly at the door rather than silently clipped every timestep.
LIMITS = {
    "cooling_offset": (-2.0, 1.5),
    "heating_offset": (-1.5, 2.0),
    "unocc_cooling_offset": (-2.0, 5.0),
    "unocc_heating_offset": (-5.0, 2.0),
    "precool_hours": (0.0, 6.0),
    "precool_depth": (0.0, 3.0),
    "peak_offset": (0.0, 3.0),
}


def validate(raw: dict) -> Policy:
    """Turn a dict from the LLM into a Policy, or raise with a fixable reason.

    The error messages are written to be read by the model, not by a human --
    each one says what was wrong AND what the allowed range is, so the retry has
    everything it needs to succeed.
    """
    if not isinstance(raw, dict):
        raise PolicyRejected(f"Expected a JSON object, got {type(raw).__name__}.")

    unknown = set(raw) - set(Policy.__dataclass_fields__)
    if unknown:
        raise PolicyRejected(
            f"Unknown field(s): {sorted(unknown)}. "
            f"Allowed fields: {sorted(Policy.__dataclass_fields__)}."
        )

    clean = {}
    for name, (lo, hi) in LIMITS.items():
        if name not in raw:
            continue
        try:
            v = float(raw[name])
        except (TypeError, ValueError):
            raise PolicyRejected(f"Field '{name}' must be a number, got {raw[name]!r}.")
        if not lo <= v <= hi:
            raise PolicyRejected(
                f"Field '{name}' = {v} is outside the allowed range [{lo}, {hi}]. "
                f"Propose a value within that range."
            )
        clean[name] = v

    for name in ("peak_start", "peak_end"):
        if name in raw:
            try:
                v = int(raw[name])
            except (TypeError, ValueError):
                raise PolicyRejected(f"Field '{name}' must be an integer hour 0-23.")
            if not 0 <= v <= 23:
                raise PolicyRejected(f"Field '{name}' = {v} must be an hour in 0-23.")
            clean[name] = v

    if clean.get("peak_start", 17) >= clean.get("peak_end", 21):
        raise PolicyRejected(
            f"peak_start ({clean.get('peak_start', 17)}) must be strictly less than "
            f"peak_end ({clean.get('peak_end', 21)})."
        )

    if "zones" in raw and raw["zones"] is not None:
        zs = [str(z).upper() for z in raw["zones"]]
        bad = [z for z in zs if z not in C.ZONES]
        if bad:
            raise PolicyRejected(
                f"Unknown zone(s): {bad}. Valid zones are: {C.ZONES}."
            )
        clean["zones"] = zs

    if "reason" in raw:
        clean["reason"] = str(raw["reason"])[:300]

    return Policy(**clean)


def target_setpoints(policy: Policy, zone: str, hour: float,
                     base_cool: float, base_heat: float):
    """Resolve a policy into concrete setpoints for one zone at one instant.

    Pure function of (policy, hour, base setpoints) -- no I/O, no state. That
    makes it trivially unit-testable, which matters because this runs 35,000
    times per simulation and a bug here is a silently wrong result.
    """
    if policy.zones is not None and zone not in policy.zones:
        return base_cool, base_heat

    occupied = C.OCCUPIED_START <= hour < C.OCCUPIED_END
    if occupied:
        cool = base_cool + policy.cooling_offset
        heat = base_heat + policy.heating_offset
    else:
        cool = base_cool + policy.unocc_cooling_offset
        heat = base_heat + policy.unocc_heating_offset

    precool_start = policy.peak_start - policy.precool_hours
    if policy.precool_hours > 0 and precool_start <= hour < policy.peak_start:
        cool -= policy.precool_depth
    elif policy.peak_start <= hour < policy.peak_end:
        cool += policy.peak_offset

    return cool, heat


def clamp(cool: float, heat: float, prev_cool=None, prev_heat=None,
          dt_hours: float = 0.25):
    """Final hard safety clamp. Nothing reaches an actuator without passing here.

    Applied every timestep regardless of how the value was produced, so even a
    validator bug cannot push the building outside the envelope.
    """
    cool = min(max(cool, C.COOLING_SP_MIN), C.COOLING_SP_MAX)
    heat = min(max(heat, C.HEATING_SP_MIN), C.HEATING_SP_MAX)

    # Preserve the deadband by pushing heating down, never cooling down: cooling
    # is the expensive one in a hot climate and we would rather not spend there.
    if cool - heat < C.MIN_DEADBAND:
        heat = cool - C.MIN_DEADBAND

    # Rate limit. Slamming a setpoint 3 degC causes a compressor surge that shows
    # up as a peak-demand spike -- exactly the thing we are claiming to reduce.
    max_step = C.MAX_SP_RAMP * dt_hours
    if prev_cool is not None:
        cool = min(max(cool, prev_cool - max_step), prev_cool + max_step)
    if prev_heat is not None:
        heat = min(max(heat, prev_heat - max_step), prev_heat + max_step)

    return cool, heat
