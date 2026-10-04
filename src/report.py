"""Markdown report export for the personal footprint passport."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .calculator import Footprint, equivalents
from .utils import fmt, fmt_co2, fmt_energy, fmt_water


def build_markdown_report(
    total: Footprint,
    breakdown: pd.DataFrame,
    region: str,
    model_label: str,
    reasoning_share: float,
    insights: list[str],
) -> str:
    lines = [
        "# My Annual GenAI Footprint Passport",
        f"_Generated {date.today().isoformat()} with GenAI Footprint Lab_",
        "",
        f"- **Region:** {region}",
        f"- **Model class:** {model_label}",
        f"- **Share of requests using reasoning mode:** {reasoning_share:.0%}",
        "",
        "## Totals (per year)",
        f"- Energy (incl. data-centre overhead): **{fmt_energy(total.facility_wh)}**",
        f"- Carbon: **{fmt_co2(total.co2_g)}**",
        f"- Water: **{fmt_water(total.water_ml)}**",
        "",
        "## In everyday terms",
    ]
    for e in equivalents(total):
        lines.append(f"- {e['icon']} {fmt(e['value'])} {e['label']}")
    lines += ["", "## Breakdown by activity", "", breakdown.to_markdown(index=False) if _has_tabulate() else breakdown.to_string(index=False)]
    if insights:
        lines += ["", "## What would reduce it"]
        lines += [f"- {i}" for i in insights]
    lines += [
        "",
        "> Estimates are order-of-magnitude and depend on editable assumptions. See the Methodology tab.",
    ]
    return "\n".join(lines)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False
