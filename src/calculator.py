"""Core footprint maths: tokens -> energy -> carbon and water.

energy_it  = work x energy-per-unit            (IT equipment only)
energy_fac = energy_it x PUE                   (adds cooling, power delivery)
CO2e       = energy_fac x grid carbon intensity
water      = energy_it x WUE  +  energy_fac x EWIF   (on-site + off-site)
"""

from __future__ import annotations

from dataclasses import dataclass

from .data import (
    BOTTLE_ML,
    CAR_G_CO2_PER_KM,
    INPUT_TOKEN_WEIGHT,
    LED_BULB_WATTS,
    MODEL_CLASSES,
    PHONE_CHARGE_WH,
    REASONING_TOKEN_MULTIPLIER,
    REGIONS,
    SHOWER_LITRES,
    TASKS,
    TREE_KG_CO2_PER_YEAR,
)


@dataclass
class Footprint:
    it_wh: float = 0.0        # IT equipment energy
    facility_wh: float = 0.0  # incl. PUE overhead
    co2_g: float = 0.0
    water_ml: float = 0.0

    def __add__(self, other: "Footprint") -> "Footprint":
        return Footprint(
            self.it_wh + other.it_wh,
            self.facility_wh + other.facility_wh,
            self.co2_g + other.co2_g,
            self.water_ml + other.water_ml,
        )

    def scale(self, k: float) -> "Footprint":
        return Footprint(self.it_wh * k, self.facility_wh * k, self.co2_g * k, self.water_ml * k)


def text_energy_wh(
    model_key: str,
    in_tokens: float,
    out_tokens: float,
    reasoning: bool = False,
) -> float:
    """IT energy (Wh) for one text request."""
    wh_1k = MODEL_CLASSES[model_key]["text_wh_per_1k_tokens"]
    out = out_tokens * (REASONING_TOKEN_MULTIPLIER if reasoning else 1.0)
    return wh_1k * (out + INPUT_TOKEN_WEIGHT * in_tokens) / 1000.0


def task_energy_wh(
    task_key: str,
    model_key: str,
    reasoning: bool = False,
    token_scale: float = 1.0,
) -> float:
    """IT energy (Wh) for one unit of a catalogue task."""
    task = TASKS[task_key]
    kind = task["kind"]
    if kind == "text":
        return text_energy_wh(
            model_key,
            task["in"] * token_scale,
            task["out"] * token_scale,
            reasoning,
        )
    if kind == "image":
        return MODEL_CLASSES[model_key]["image_wh"]
    if kind == "video":
        return MODEL_CLASSES[model_key]["video_wh"]
    raise ValueError(f"Unknown task kind: {kind}")


def footprint_from_energy(it_wh: float, region_key: str, params: dict | None = None) -> Footprint:
    """Convert IT energy into facility energy, carbon and water for a region.

    `params` may override ci / pue / wue / ewif (used by the Monte Carlo lab).
    """
    r = dict(REGIONS[region_key])
    if params:
        r.update(params)
    facility_wh = it_wh * r["pue"]
    co2_g = facility_wh / 1000.0 * r["ci"]
    water_l = it_wh / 1000.0 * r["wue"] + facility_wh / 1000.0 * r["ewif"]
    return Footprint(it_wh, facility_wh, co2_g, water_l * 1000.0)


def footprint_for_task(
    task_key: str,
    model_key: str,
    region_key: str,
    n: float = 1.0,
    reasoning: bool = False,
    token_scale: float = 1.0,
) -> Footprint:
    wh = task_energy_wh(task_key, model_key, reasoning, token_scale) * n
    return footprint_from_energy(wh, region_key)


def call_footprint(
    input_tokens: float,
    output_tokens: float,
    model_key: str,
    region_key: str,
) -> Footprint:
    """Footprint of one real LLM call, from the token counts the API reported."""
    return footprint_from_energy(text_energy_wh(model_key, input_tokens, output_tokens), region_key)


def passport_footprint(
    weekly_counts: dict[str, float],
    model_key: str,
    region_key: str,
    reasoning_share: float = 0.0,
    weeks: int = 52,
) -> tuple[Footprint, list[dict]]:
    """Annual footprint for a person's weekly usage; returns (total, per-task rows)."""
    total = Footprint()
    rows: list[dict] = []
    for task_key, per_week in weekly_counts.items():
        if per_week <= 0:
            continue
        n = per_week * weeks
        if TASKS[task_key]["kind"] == "text":
            fp = footprint_for_task(task_key, model_key, region_key, n * (1 - reasoning_share)) + \
                footprint_for_task(task_key, model_key, region_key, n * reasoning_share, reasoning=True)
        else:
            fp = footprint_for_task(task_key, model_key, region_key, n)
        total = total + fp
        rows.append(
            {
                "Activity": TASKS[task_key]["label"],
                "Per week": per_week,
                "Energy (kWh/yr)": fp.facility_wh / 1000.0,
                "CO2e (kg/yr)": fp.co2_g / 1000.0,
                "Water (L/yr)": fp.water_ml / 1000.0,
            }
        )
    return total, rows


def equivalents(fp: Footprint) -> list[dict]:
    """Translate a footprint into everyday comparisons."""
    tree_day_g = TREE_KG_CO2_PER_YEAR * 1000.0 / 365.0
    return [
        {"icon": "📱", "value": fp.facility_wh / PHONE_CHARGE_WH, "label": "smartphone charges"},
        {"icon": "💡", "value": fp.facility_wh / LED_BULB_WATTS, "label": "hours of an LED bulb"},
        {"icon": "🚗", "value": fp.co2_g / CAR_G_CO2_PER_KM, "label": "km driven in a petrol car"},
        {"icon": "🌳", "value": fp.co2_g / tree_day_g, "label": "tree-days of CO2 absorption"},
        {"icon": "🥤", "value": fp.water_ml / BOTTLE_ML, "label": "500 mL water bottles"},
        {"icon": "🚿", "value": fp.water_ml / (SHOWER_LITRES * 1000.0), "label": "showers (60 L each)"},
    ]
