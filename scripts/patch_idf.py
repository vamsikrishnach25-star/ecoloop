"""Patch a DOE prototype IDF for Eco-Loop.

Three edits, all necessary:
  1. Collapse the model's multiple RunPeriod objects into ONE. Each RunPeriod is a
     separate EnergyPlus "environment" -- with three of them your callbacks fire
     three times over and the baseline/agent comparison stops being apples to apples.
  2. Turn on the Fanger PMV thermal comfort model on every People object. The DOE
     prototypes ship with it OFF, so "Zone Thermal Comfort Fanger Model PMV" simply
     does not exist as a variable until you do this. 20% of the grade depends on it.
  3. Force the timestep so the control cadence is known.

Usage:
    python scripts/patch_idf.py models/baseline.idf models/sim.idf \
        --start 7 1 --end 7 14 --timestep 4
"""

import argparse
import re
import sys
from pathlib import Path

RUNPERIOD_TEMPLATE = """
  RunPeriod,
    ECOLOOP_PERIOD,          !- Name
    {sm},                    !- Begin Month
    {sd},                    !- Begin Day of Month
    ,                        !- Begin Year
    {em},                    !- End Month
    {ed},                    !- End Day of Month
    ,                        !- End Year
    Sunday,                  !- Day of Week for Start Day
    No,                      !- Use Weather File Holidays and Special Days
    No,                      !- Use Weather File Daylight Saving Period
    No,                      !- Apply Weekend Holiday Rule
    Yes,                     !- Use Weather File Rain Indicators
    Yes,                     !- Use Weather File Snow Indicators
    No;                      !- Treat Weather as Actual
"""


def iter_objects(text, obj_type):
    """Yield (start, end, body) for each object of obj_type at top level."""
    pattern = re.compile(r"^[ \t]*" + re.escape(obj_type) + r"\s*,.*?;", re.S | re.M | re.I)
    for m in pattern.finditer(text):
        yield m.start(), m.end(), m.group(0)


def strip_comments(body):
    """Remove '!- ...' end-of-line comments.

    This MUST happen before splitting on commas. In IDF the comment trails the
    comma, so a naive split leaves each chunk looking like
    '   !- Zone Name\\n    BLDG_OCC_SCH' -- take text before '!' and you get the
    empty string for every single field.
    """
    return "\n".join(line.split("!")[0] for line in body.splitlines())


def split_fields(body):
    """Split an IDF object body into its comma-separated fields."""
    clean = strip_comments(body).strip().rstrip(";")
    return [f.strip() for f in clean.split(",")]


def replace_run_periods(text, start, end):
    spans = [(s, e) for s, e, _ in iter_objects(text, "RunPeriod")]
    if not spans:
        print("  ! no RunPeriod found -- leaving as is", file=sys.stderr)
        return text, 0
    new = RUNPERIOD_TEMPLATE.format(sm=start[0], sd=start[1], em=end[0], ed=end[1])
    # Replace back-to-front so earlier offsets stay valid.
    for i, (s, e) in enumerate(reversed(spans)):
        text = text[:s] + (new if i == len(spans) - 1 else "") + text[e:]
    return text, len(spans)


def enable_pmv(text):
    """Set 'Thermal Comfort Model 1 Type' to Fanger on every People object.

    People field order (EnergyPlus 24.x), 1-indexed after the object name field:
      0 People, 1 Name, 2 Zone Name, 3 Number of People Schedule Name,
      4 Number of People Calculation Method, 5 Number of People,
      6 People per Floor Area, 7 Floor Area per Person, 8 Fraction Radiant,
      9 Sensible Heat Fraction, 10 Activity Level Schedule Name,
      11 CO2 Generation Rate, 12 Enable ASHRAE 55 Comfort Warnings,
      13 Mean Radiant Temperature Calculation Type, 14 Surface Name/Angle Factor List,
      15 Work Efficiency Schedule Name, 16 Clothing Insulation Calculation Method,
      17 Clothing Insulation Calculation Method Schedule Name,
      18 Clothing Insulation Schedule Name, 19 Air Velocity Schedule Name,
      20 Thermal Comfort Model 1 Type
    """
    count = 0
    out = text
    for s, e, body in list(iter_objects(text, "People"))[::-1]:
        fields = split_fields(body)
        # Pad out to at least 21 fields so index 20 exists.
        while len(fields) < 21:
            fields.append("")
        if fields[20].lower() == "fanger":
            continue
        fields[20] = "Fanger"
        # A Fanger calculation needs schedules for work efficiency, clothing and air
        # velocity. Fall back to the model's own schedules if the fields are blank.
        if not fields[15]:
            fields[15] = "Work_Eff_Sch"
        if not fields[16]:
            fields[16] = "ClothingInsulationSchedule"
        if not fields[18]:
            fields[18] = "Clothing_Sch"
        if not fields[19]:
            fields[19] = "Air_Velo_Sch"
        rebuilt = "  People,\n" + ",\n".join("    " + f for f in fields[1:]) + ";\n"
        out = out[:s] + rebuilt + out[e:]
        count += 1
    return out, count


def ensure_comfort_schedules(text):
    """Add the constant schedules the Fanger model needs, if absent."""
    needed = {
        "Work_Eff_Sch": 0.0,
        "Clothing_Sch": 0.5,
        "Air_Velo_Sch": 0.137,
    }
    additions = []
    lowered = text.lower()
    for name, value in needed.items():
        if f"{name.lower()}," not in lowered:
            additions.append(
                f"\n  Schedule:Constant,\n    {name},\n    Any Number,\n    {value};\n"
            )
    if "clothinginsulationschedule," not in lowered:
        additions.append(
            "\n  Schedule:Constant,\n    ClothingInsulationSchedule,\n"
            "    Any Number,\n    1;\n"
        )
    if "any number," not in lowered:
        additions.append("\n  ScheduleTypeLimits,\n    Any Number;\n")
    return text + "".join(additions), len(additions)


def set_timestep(text, n):
    if re.search(r"^\s*Timestep\s*,", text, re.M | re.I):
        return re.sub(r"^\s*Timestep\s*,\s*\d+\s*;", f"  Timestep,{n};", text,
                      flags=re.M | re.I)
    return text + f"\n  Timestep,{n};\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--start", nargs=2, type=int, default=[7, 1], metavar=("MONTH", "DAY"))
    p.add_argument("--end", nargs=2, type=int, default=[7, 14], metavar=("MONTH", "DAY"))
    p.add_argument("--timestep", type=int, default=4)
    args = p.parse_args()

    text = Path(args.src).read_text()

    text, n_rp = replace_run_periods(text, args.start, args.end)
    print(f"  run periods: {n_rp} -> 1  "
          f"({args.start[0]}/{args.start[1]} to {args.end[0]}/{args.end[1]})")

    text, n_sched = ensure_comfort_schedules(text)
    text, n_people = enable_pmv(text)
    print(f"  PMV enabled on {n_people} People objects (+{n_sched} schedules)")

    text = set_timestep(text, args.timestep)
    print(f"  timestep: {args.timestep}/hour")

    Path(args.dst).write_text(text)
    print(f"  wrote {args.dst}")


if __name__ == "__main__":
    main()
