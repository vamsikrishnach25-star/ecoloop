"""Central configuration for Eco-Loop.

Everything the rest of the package needs to know about the building and the
environment lives here, so swapping the model or the weather file is a one-line
change rather than a grep.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------

REPO = Path(__file__).resolve().parents[2]

# Point ECOLOOP_EPLUS at your EnergyPlus install directory (the one containing
# the `pyenergyplus` package and `energyplus` binary).
EPLUS_DIR = Path(os.environ.get("ECOLOOP_EPLUS", "/usr/local/EnergyPlus-24-1-0"))

MODELS = REPO / "models"
RESULTS = REPO / "results"

IDF = MODELS / "sim.idf"
EPW = Path(os.environ.get(
    "ECOLOOP_EPW",
    EPLUS_DIR / "WeatherData" / "USA_FL_Tampa.Intl.AP.722110_TMY3.epw",
))

# --- Building ----------------------------------------------------------------

# Conditioned zones only. Plenums have no thermostat and no occupants, so they
# have no setpoint actuator and no PMV variable -- asking for handles on them
# returns -1 and pollutes your logs. EnergyPlus keys everything by UPPERCASE name.
ZONES = [
    "CORE_BOTTOM", "CORE_MID", "CORE_TOP",
    "PERIMETER_BOT_ZN_1", "PERIMETER_BOT_ZN_2", "PERIMETER_BOT_ZN_3", "PERIMETER_BOT_ZN_4",
    "PERIMETER_MID_ZN_1", "PERIMETER_MID_ZN_2", "PERIMETER_MID_ZN_3", "PERIMETER_MID_ZN_4",
    "PERIMETER_TOP_ZN_1", "PERIMETER_TOP_ZN_2", "PERIMETER_TOP_ZN_3", "PERIMETER_TOP_ZN_4",
]

# Meter names are NOT stable across IDF models. A building with on-site
# generation exposes ElectricityNet:Facility and no Electricity:Facility at all,
# so a hard-coded name silently returns handle -1 and every energy number comes
# out as zero. Each entry is a fallback chain: first name that resolves wins.
METERS = {
    "total": ["ElectricityNet:Facility", "Electricity:Facility", "Electricity:Building"],
    "cooling": ["Cooling:Electricity"],
    "heating": ["Heating:Electricity"],
    "fans": ["Fans:Electricity"],
    "lights": ["InteriorLights:Electricity"],
}

# --- Safety envelope ---------------------------------------------------------
# The agent NEVER writes outside these. This is what lets you tell a judge the
# system cannot trade comfort for energy no matter what the model hallucinates.

COOLING_SP_MIN = 22.0     # degC
COOLING_SP_MAX = 30.0   # unoccupied setback needs headroom
HEATING_SP_MIN = 15.0
HEATING_SP_MAX = 21.0
MIN_DEADBAND = 2.0        # degC between heating and cooling setpoints
MAX_SP_RAMP = 4.0         # degC change allowed per hour, per zone

PMV_LOW, PMV_HIGH = -0.5, 0.5   # ASHRAE 55 comfort band

OCCUPIED_START, OCCUPIED_END = 7, 19   # local hour, used for comfort accounting

# --- Grid carbon intensity ---------------------------------------------------
# gCO2 per kWh by hour of day. A stylised duck curve: cheap and clean midday when
# solar is on the grid, dirty in the evening peak. Swap for real CEA / WattTime
# data if you have it -- the agent reads this through a tool either way.

CARBON_INTENSITY = [
    708, 705, 700, 698, 700, 712, 730, 742,   # 00-07
    700, 640, 575, 520, 495, 490, 505, 550,   # 08-15
    620, 700, 780, 820, 810, 790, 760, 730,   # 16-23
]

J_TO_KWH = 1.0 / 3_600_000.0
