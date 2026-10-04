"""Small formatting helpers shared across the app."""

from __future__ import annotations


def fmt(x: float) -> str:
    """Human-friendly number formatting."""
    ax = abs(x)
    if ax == 0:
        return "0"
    if ax >= 1000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    if ax >= 1:
        return f"{x:.2f}"
    return f"{x:.3g}"


def fmt_energy(wh: float) -> str:
    if abs(wh) >= 1e6:
        return f"{fmt(wh / 1e6)} MWh"
    if abs(wh) >= 1e3:
        return f"{fmt(wh / 1e3)} kWh"
    return f"{fmt(wh)} Wh"


def fmt_co2(g: float) -> str:
    if abs(g) >= 1e6:
        return f"{fmt(g / 1e6)} t CO2e"
    if abs(g) >= 1e3:
        return f"{fmt(g / 1e3)} kg CO2e"
    return f"{fmt(g)} g CO2e"


def fmt_water(ml: float) -> str:
    if abs(ml) >= 1e6:
        return f"{fmt(ml / 1e6)} m3"
    if abs(ml) >= 1e3:
        return f"{fmt(ml / 1e3)} L"
    return f"{fmt(ml)} mL"
