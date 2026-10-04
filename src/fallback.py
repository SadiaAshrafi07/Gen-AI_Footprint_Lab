"""Deterministic fallbacks used in Demo mode (no API key).

They give a faithful, rule-based version of each AI Studio feature so the app
still works for reviewers without a key. Every output is labelled as demo.
"""

from __future__ import annotations

from .calculator import footprint_for_task
from .data import MODEL_CLASSES, REGIONS
from .prompts import org_to_text
from .utils import fmt, fmt_co2, fmt_water

DEMO_NOTE = "_Demo mode - rule-based answer. Add an API key in the sidebar for generative AI answers._"


def demo_advisor(question: str, region: str, passport: dict | None, org: dict | None) -> str:
    q = question.lower()
    parts: list[str] = []

    if any(w in q for w in ("region", "grid", "where", "country", "india", "norway", "location")):
        ranked = sorted(REGIONS, key=lambda k: footprint_for_task("detailed_chat", "medium", k).co2_g)
        best, worst = ranked[0], ranked[-1]
        fb, fw = footprint_for_task("detailed_chat", "medium", best), footprint_for_task("detailed_chat", "medium", worst)
        parts.append(
            f"For one chat answer on a medium model, {best} emits {fmt_co2(fb.co2_g)} versus {fmt_co2(fw.co2_g)} in "
            f"{worst} (~{fmt(fw.co2_g / fb.co2_g)}x). Check the Region & Timing tab for the carbon-vs-water trade-off."
        )
    if any(w in q for w in ("water", "cool", "litre", "liter")):
        fp = footprint_for_task("detailed_chat", "medium", region)
        parts.append(
            f"In {region}, one chat answer uses about {fmt_water(fp.water_ml)} (cooling plus electricity generation)."
        )
    if any(w in q for w in ("image", "video", "picture")):
        c = footprint_for_task("detailed_chat", "medium", region)
        i = footprint_for_task("image", "medium", region)
        v = footprint_for_task("video", "medium", region)
        parts.append(
            f"On a medium model an image is ~{fmt(i.co2_g / c.co2_g)}x and a 5-second video ~{fmt(v.co2_g / c.co2_g)}x "
            "the carbon of a chat answer."
        )
    if any(w in q for w in ("model", "small", "large", "size", "frontier")):
        s = footprint_for_task("detailed_chat", "small", region)
        l = footprint_for_task("detailed_chat", "large", region)
        parts.append(
            f"Switching a chat answer from a {MODEL_CLASSES['large']['label']} to a {MODEL_CLASSES['small']['label']} "
            f"model cuts carbon by ~{(1 - s.co2_g / l.co2_g) * 100:.0f}%."
        )
    if passport and any(w in q for w in ("my", "reduce", "cut", "lower", "save", "footprint", "biggest", "first")):
        parts.append("From your passport: " + " ".join(passport.get("insights", [])))
    if org and any(w in q for w in ("org", "company", "team", "lever", "plan", "employees")):
        parts.append(org_to_text(org))
    if not parts:
        parts.append(
            "Biggest levers, in general order of impact: use the smallest model that works, avoid unnecessary "
            "image/video generation and reasoning mode, ask for concise answers, cache repeated requests, and run "
            "workloads on cleaner grids or at cleaner hours."
        )
    return "\n\n".join(parts) + "\n\n" + DEMO_NOTE


def demo_summary(org: dict) -> str:
    deltas = []
    prev = org["waterfall"][0]["co2_t"]
    for s in org["waterfall"][1:]:
        deltas.append((s["step"], prev - s["co2_t"]))
        prev = s["co2_t"]
    deltas.sort(key=lambda x: x[1], reverse=True)
    return (
        f"**Headline:** Planned levers cut {org['years']}-year AI carbon by {org['co2_cut']:.0f}% "
        f"({fmt(org['base_co2'])} t to {fmt(org['opt_co2'])} t CO2e).\n\n"
        f"- Water falls {org['water_cut']:.0f}% ({fmt(org['base_water'])} m3 to {fmt(org['opt_water'])} m3).\n"
        f"- Largest single lever: {deltas[0][0]} (about {fmt(deltas[0][1])} t CO2e).\n"
        f"- Region: {org['region']}, {org['employees']:,} employees, {org['adoption']:.0%} adoption.\n\n"
        "**Actions (by impact):** "
        + "; ".join(f"{i + 1}. {d[0]}" for i, d in enumerate(deltas[:3]))
        + ".\n\n**Caveat:** figures are order-of-magnitude estimates; see the Uncertainty Lab.\n\n"
        + DEMO_NOTE
    )
