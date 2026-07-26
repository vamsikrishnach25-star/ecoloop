"""Unit tests for the safety layer. No EnergyPlus needed -- runs in <1s.

These matter more than they look. `clamp` executes 15 zones x 1344 timesteps per
run; a sign error here produces a plausible-looking results table that is wrong.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from ecoloop import config as C
from ecoloop.policy import Policy, validate, clamp, target_setpoints, PolicyRejected


def test_clamp_respects_hard_envelope():
    cool, heat = clamp(99.0, -99.0)
    assert cool == C.COOLING_SP_MAX
    assert heat == C.HEATING_SP_MIN


def test_clamp_preserves_deadband():
    cool, heat = clamp(24.0, 23.5)
    assert cool - heat >= C.MIN_DEADBAND
    assert cool == 24.0, "cooling must not be dragged down; heating gives way"


def test_clamp_rate_limits():
    cool, _ = clamp(30.0, 20.0, prev_cool=24.0, prev_heat=20.0, dt_hours=0.25)
    assert cool <= 24.0 + C.MAX_SP_RAMP * 0.25 + 1e-9


def test_validator_rejects_out_of_range():
    with pytest.raises(PolicyRejected, match="outside the allowed range"):
        validate({"cooling_offset": 50})


def test_validator_rejects_unknown_field():
    with pytest.raises(PolicyRejected, match="Unknown field"):
        validate({"turn_off_hvac": True})


def test_validator_rejects_bad_zone():
    with pytest.raises(PolicyRejected, match="Unknown zone"):
        validate({"zones": ["THE_ROOF"]})


def test_validator_error_names_the_fix():
    """Every rejection must tell the model the allowed range, or it cannot self-correct."""
    try:
        validate({"precool_depth": 99})
    except PolicyRejected as e:
        assert "range" in str(e) and "0.0" in str(e)


def test_occupied_and_unoccupied_offsets_are_separate():
    p = Policy(cooling_offset=0.5, unocc_cooling_offset=4.0)
    occ, _ = target_setpoints(p, "CORE_MID", 12.0, 24.0, 21.0)
    unocc, _ = target_setpoints(p, "CORE_MID", 3.0, 26.7, 15.6)
    assert occ == pytest.approx(24.5)
    assert unocc == pytest.approx(30.7)


def test_precool_lowers_setpoint_before_peak():
    p = Policy(precool_hours=3, precool_depth=2.0, peak_start=17, peak_offset=1.0)
    pre, _ = target_setpoints(p, "CORE_MID", 15.0, 24.0, 21.0)
    during, _ = target_setpoints(p, "CORE_MID", 18.0, 24.0, 21.0)
    assert pre == pytest.approx(22.0)
    assert during == pytest.approx(25.0)


def test_zone_scoping():
    p = Policy(cooling_offset=3.0, zones=["CORE_MID"])
    touched, _ = target_setpoints(p, "CORE_MID", 12.0, 24.0, 21.0)
    untouched, _ = target_setpoints(p, "CORE_TOP", 12.0, 24.0, 21.0)
    assert touched == pytest.approx(27.0)
    assert untouched == pytest.approx(24.0)
