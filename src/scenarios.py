"""Scenario engines: organisation planner, uncertainty (Monte Carlo) and timing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calculator import Footprint, footprint_for_task, footprint_from_energy, task_energy_wh
from .data import REGIONS


# --------------------------------------------------------------------------- #
# Organisation planner
# --------------------------------------------------------------------------- #
@dataclass
class OrgParams:
    employees: int = 5000
    adoption: float = 0.6            # share of employees using GenAI
    queries_per_day: float = 20.0    # text queries per active user per workday
    images_per_day: float = 0.5      # images per active user per workday
    workdays: int = 250
    growth: float = 0.35             # yearly growth in usage
    years: int = 3
    region: str = "India (national grid)"
    shares: dict = field(default_factory=lambda: {"small": 0.2, "medium": 0.3, "large": 0.5})
    reasoning_share: float = 0.1


@dataclass
class Levers:
    cache_pct: float = 0.0           # queries avoided through caching / dedupe
    trim_pct: float = 0.0            # tokens saved through prompt + output discipline
    rightsize_pct: float = 0.0       # share of large-model traffic moved to small models
    region_target: str = "Norway (hydro)"
    region_shift_pct: float = 0.0    # share of workload moved to a cleaner region


def _yearly_footprint(
    p: OrgParams,
    year_index: int,
    lv: Levers,
) -> Footprint:
    active = p.employees * p.adoption * (1 + p.growth) ** year_index
    queries = active * p.queries_per_day * p.workdays * (1.0 - lv.cache_pct)
    images = active * p.images_per_day * p.workdays * (1.0 - lv.cache_pct)
    token_scale = 1.0 - lv.trim_pct

    shares = dict(p.shares)
    moved = shares["large"] * lv.rightsize_pct
    shares["large"] -= moved
    shares["small"] += moved

    def at_region(region: str) -> Footprint:
        total = Footprint()
        for m, s in shares.items():
            # reasoning applies to a fraction of text queries
            base = footprint_for_task("detailed_chat", m, region, queries * s * (1 - p.reasoning_share),
                                      reasoning=False, token_scale=token_scale)
            reas = footprint_for_task("detailed_chat", m, region, queries * s * p.reasoning_share,
                                      reasoning=True, token_scale=token_scale)
            img = footprint_for_task("image", m, region, images * s)
            total = total + base + reas + img
        return total

    own = at_region(p.region)
    if lv.region_shift_pct > 0:
        tgt = at_region(lv.region_target)
        return own.scale(1 - lv.region_shift_pct) + tgt.scale(lv.region_shift_pct)
    return own


def project_org(p: OrgParams, lv: Levers) -> pd.DataFrame:
    """Year-by-year footprint for a set of levers."""
    rows = []
    for y in range(p.years):
        fp = _yearly_footprint(p, y, lv)
        rows.append(
            {
                "year": y + 1,
                "energy_mwh": fp.facility_wh / 1e6,
                "co2_t": fp.co2_g / 1e6,
                "water_m3": fp.water_ml / 1e6,
            }
        )
    return pd.DataFrame(rows)


def lever_waterfall(p: OrgParams, lv: Levers) -> pd.DataFrame:
    """Cumulative CO2 (tonnes) after enabling levers one at a time."""
    steps = [
        ("Baseline", Levers(0, 0, 0, lv.region_target, 0)),
        ("Caching", Levers(lv.cache_pct, 0, 0, lv.region_target, 0)),
        ("Token discipline", Levers(lv.cache_pct, lv.trim_pct, 0, lv.region_target, 0)),
        ("Right-sizing", Levers(lv.cache_pct, lv.trim_pct, lv.rightsize_pct, lv.region_target, 0)),
        ("Cleaner region", lv),
    ]
    rows = []
    prev = None
    for name, lev in steps:
        df = project_org(p, lev)
        co2, water = df["co2_t"].sum(), df["water_m3"].sum()
        rows.append(
            {
                "step": name,
                "co2_t": co2,
                "water_m3": water,
                "co2_delta": 0.0 if prev is None else co2 - prev[0],
                "water_delta": 0.0 if prev is None else water - prev[1],
            }
        )
        prev = (co2, water)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Uncertainty lab
# --------------------------------------------------------------------------- #
def monte_carlo(
    task_key: str,
    model_key: str,
    region_key: str,
    n_queries: int = 1000,
    draws: int = 5000,
    reasoning: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample uncertain parameters and return the distribution of outcomes."""
    rng = np.random.default_rng(seed)
    r = REGIONS[region_key]
    base_wh = task_energy_wh(task_key, model_key, reasoning) * n_queries

    energy_mult = rng.lognormal(mean=0.0, sigma=0.5, size=draws)
    pue = rng.triangular(max(1.05, r["pue"] * 0.9), r["pue"], r["pue"] * 1.25, draws)
    ci = rng.triangular(r["ci"] * 0.8, r["ci"], r["ci"] * 1.2, draws)
    wue = rng.triangular(r["wue"] * 0.5, r["wue"], r["wue"] * 1.6, draws)
    ewif = rng.triangular(r["ewif"] * 0.7, r["ewif"], r["ewif"] * 1.3, draws)

    it_wh = base_wh * energy_mult
    fac_wh = it_wh * pue
    co2_kg = fac_wh / 1000.0 * ci / 1000.0
    water_l = it_wh / 1000.0 * wue + fac_wh / 1000.0 * ewif
    return pd.DataFrame(
        {"energy_kwh": fac_wh / 1000.0, "co2_kg": co2_kg, "water_l": water_l}
    )


def summarise_distribution(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append(
            {
                "metric": col,
                "P5": df[col].quantile(0.05),
                "median": df[col].median(),
                "P95": df[col].quantile(0.95),
                "P95 / P5": df[col].quantile(0.95) / max(df[col].quantile(0.05), 1e-12),
            }
        )
    return pd.DataFrame(rows)


def sensitivity(
    task_key: str,
    model_key: str,
    region_key: str,
    reasoning: bool = False,
) -> dict[str, pd.DataFrame]:
    """One-at-a-time sensitivity (tornado) for CO2 and water, % change vs base."""
    r = REGIONS[region_key]
    base_wh = task_energy_wh(task_key, model_key, reasoning)
    base = footprint_from_energy(base_wh, region_key)

    cases = {
        "Energy per query (x0.5 / x2)": ("energy", 0.5, 2.0),
        "PUE (-10% / +25%)": ("pue", r["pue"] * 0.9, r["pue"] * 1.25),
        "Grid carbon intensity (-20% / +20%)": ("ci", r["ci"] * 0.8, r["ci"] * 1.2),
        "On-site water WUE (-50% / +60%)": ("wue", r["wue"] * 0.5, r["wue"] * 1.6),
        "Electricity water EWIF (-30% / +30%)": ("ewif", r["ewif"] * 0.7, r["ewif"] * 1.3),
    }

    def run(kind: str, value: float) -> Footprint:
        if kind == "energy":
            return footprint_from_energy(base_wh * value, region_key)
        return footprint_from_energy(base_wh, region_key, {kind: value})

    co2_rows, water_rows = [], []
    for label, (kind, lo, hi) in cases.items():
        f_lo, f_hi = run(kind, lo), run(kind, hi)
        co2_rows.append({"parameter": label,
                         "low": (f_lo.co2_g / base.co2_g - 1) * 100,
                         "high": (f_hi.co2_g / base.co2_g - 1) * 100})
        water_rows.append({"parameter": label,
                           "low": (f_lo.water_ml / base.water_ml - 1) * 100,
                           "high": (f_hi.water_ml / base.water_ml - 1) * 100})
    co2_df = pd.DataFrame(co2_rows)
    water_df = pd.DataFrame(water_rows)
    # drop parameters with no effect on that metric
    co2_df = co2_df[(co2_df["low"].abs() + co2_df["high"].abs()) > 1e-9]
    water_df = water_df[(water_df["low"].abs() + water_df["high"].abs()) > 1e-9]
    return {"co2": co2_df, "water": water_df}


# --------------------------------------------------------------------------- #
# Carbon-aware timing (synthetic illustrative profile)
# --------------------------------------------------------------------------- #
def diurnal_profile(base_ci: float, solar_dip: float = 0.25, evening_peak: float = 0.20) -> np.ndarray:
    """Illustrative 24h grid-intensity curve (gCO2e/kWh), mean = base_ci.

    Evening demand peak around 20:00 and a midday solar dip around 13:00.
    This is a *synthetic* shape for demonstration, not measured data.
    """
    h = np.arange(24)
    peak = evening_peak * np.cos(2 * np.pi * (h - 20) / 24)
    dip = solar_dip * np.exp(-(((h - 13) / 3.0) ** 2))
    ci = base_ci * (1 + peak - dip)
    return ci * (base_ci / ci.mean())


def timing_savings(ci_curve: np.ndarray, flexible_share: float, best_hours: int = 4) -> dict:
    """CO2 saved by moving a flexible share of a flat workload to the cleanest hours."""
    avg = float(ci_curve.mean())
    clean = float(np.sort(ci_curve)[:best_hours].mean())
    shifted = (1 - flexible_share) * avg + flexible_share * clean
    return {
        "baseline_ci": avg,
        "shifted_ci": shifted,
        "saving_pct": (1 - shifted / avg) * 100 if avg else 0.0,
        "clean_hours": sorted(int(h) for h in np.argsort(ci_curve)[:best_hours]),
    }
