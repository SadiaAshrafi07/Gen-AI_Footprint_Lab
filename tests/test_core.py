"""Unit tests. Run with:  pytest -q"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calculator import (  # noqa: E402
    footprint_for_task,
    footprint_from_energy,
    passport_footprint,
    task_energy_wh,
    text_energy_wh,
)
from src.data import MODEL_CLASSES, REGIONS, TASKS  # noqa: E402
from src.prompt_analyzer import analyse, estimate_tokens  # noqa: E402
from src.scenarios import (  # noqa: E402
    Levers,
    OrgParams,
    diurnal_profile,
    lever_waterfall,
    monte_carlo,
    project_org,
    sensitivity,
    timing_savings,
)


def test_energy_scales_with_model_size():
    small = task_energy_wh("detailed_chat", "small")
    medium = task_energy_wh("detailed_chat", "medium")
    large = task_energy_wh("detailed_chat", "large")
    assert small < medium < large


def test_reasoning_increases_energy():
    assert task_energy_wh("detailed_chat", "large", reasoning=True) > 4 * task_energy_wh("detailed_chat", "large")


def test_text_energy_formula():
    wh = text_energy_wh("medium", 1000, 1000)
    expected = MODEL_CLASSES["medium"]["text_wh_per_1k_tokens"] * (1000 + 0.15 * 1000) / 1000
    assert math.isclose(wh, expected)


def test_footprint_formula():
    r = REGIONS["USA (average)"]
    fp = footprint_from_energy(100.0, "USA (average)")
    assert math.isclose(fp.facility_wh, 100 * r["pue"])
    assert math.isclose(fp.co2_g, 100 * r["pue"] / 1000 * r["ci"])
    water_l = 100 / 1000 * r["wue"] + 100 * r["pue"] / 1000 * r["ewif"]
    assert math.isclose(fp.water_ml, water_l * 1000)


def test_cleaner_grid_lower_carbon():
    dirty = footprint_for_task("image", "medium", "India (national grid)")
    clean = footprint_for_task("image", "medium", "Norway (hydro)")
    assert clean.co2_g < dirty.co2_g


def test_all_tasks_and_regions_compute():
    for t in TASKS:
        for m in MODEL_CLASSES:
            for r in REGIONS:
                fp = footprint_for_task(t, m, r, 10)
                assert fp.co2_g > 0 and fp.water_ml > 0


def test_passport_additive():
    counts = {"quick_qa": 10, "image": 2}
    total, rows = passport_footprint(counts, "medium", "India (national grid)")
    assert len(rows) == 2
    assert math.isclose(total.co2_g / 1000, sum(r["CO2e (kg/yr)"] for r in rows), rel_tol=1e-9)


def test_prompt_analyser_trims_fillers():
    res = analyse("Please could you kindly just summarise this. Please could you kindly just summarise this.")
    assert res["filler_total"] > 0
    assert res["repeated_sentences"] >= 1
    assert res["lean_tokens"] < res["tokens"]
    assert 0 <= res["score"] <= 100


def test_estimate_tokens_empty():
    assert estimate_tokens("   ") == 0
    assert estimate_tokens("hello world") >= 2


def test_org_levers_reduce_footprint():
    p = OrgParams()
    base = project_org(p, Levers())["co2_t"].sum()
    opt = project_org(p, Levers(0.2, 0.2, 0.5, "Norway (hydro)", 0.3))["co2_t"].sum()
    assert opt < base
    wf = lever_waterfall(p, Levers(0.2, 0.2, 0.5, "Norway (hydro)", 0.3))
    assert math.isclose(wf["co2_t"].iloc[-1], opt, rel_tol=1e-9)
    assert wf["co2_t"].is_monotonic_decreasing


def test_monte_carlo_reproducible_and_positive():
    a = monte_carlo("detailed_chat", "medium", "India (national grid)", draws=500, seed=1)
    b = monte_carlo("detailed_chat", "medium", "India (national grid)", draws=500, seed=1)
    assert a.equals(b)
    assert (a > 0).all().all()


def test_sensitivity_has_both_metrics():
    s = sensitivity("detailed_chat", "medium", "USA (average)")
    assert not s["co2"].empty and not s["water"].empty
    assert "On-site water WUE (-50% / +60%)" not in set(s["co2"]["parameter"])


def test_diurnal_profile_mean_and_savings():
    ci = diurnal_profile(500)
    assert math.isclose(ci.mean(), 500, rel_tol=1e-9)
    res = timing_savings(ci, 0.5, 4)
    assert res["saving_pct"] > 0
    assert timing_savings(ci, 0.0, 4)["saving_pct"] == 0
