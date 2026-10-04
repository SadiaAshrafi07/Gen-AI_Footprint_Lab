"""Plotly chart builders (consistent palette, light theme)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TEAL = "#0F766E"
AMBER = "#D97706"
BLUE = "#2563EB"
SLATE = "#475569"
RED = "#DC2626"
GREEN = "#16A34A"
PALETTE = [TEAL, AMBER, BLUE, "#7C3AED", RED, "#0891B2", SLATE]

_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=13),
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", y=-0.2),
)


def _style(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    fig.update_layout(title=dict(text=title, x=0.0), height=height, **_LAYOUT)
    return fig


def task_by_model_bar(df: pd.DataFrame, metric: str, ylabel: str, title: str) -> go.Figure:
    """df columns: task, model, <metric>."""
    fig = go.Figure()
    for i, (model, sub) in enumerate(df.groupby("model", sort=False)):
        fig.add_bar(x=sub["task"], y=sub[metric], name=model, marker_color=PALETTE[i % len(PALETTE)])
    fig.update_layout(barmode="group", yaxis_title=ylabel, yaxis_type="log")
    return _style(fig, title)


def breakdown_donut(labels: list[str], values: list[float], title: str) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker=dict(colors=PALETTE)))
    fig.update_traces(textinfo="percent")
    return _style(fig, title, height=340)


def region_bars(df: pd.DataFrame, metric: str, ylabel: str, title: str, color: str) -> go.Figure:
    d = df.sort_values(metric, ascending=True)
    fig = go.Figure(go.Bar(x=d[metric], y=d["region"], orientation="h", marker_color=color))
    fig.update_layout(xaxis_title=ylabel)
    return _style(fig, title, height=380)


def carbon_water_scatter(df: pd.DataFrame) -> go.Figure:
    """Carbon-water trade-off across regions."""
    fig = go.Figure(
        go.Scatter(
            x=df["co2_g"],
            y=df["water_ml"],
            mode="markers+text",
            text=df["region"],
            textposition="top center",
            marker=dict(size=14, color=TEAL, opacity=0.8),
        )
    )
    fig.update_layout(xaxis_title="CO2e (g)", yaxis_title="Water (mL)")
    return _style(fig, "Carbon vs water: the cleanest grid is not always the driest", height=420)


def diurnal_chart(ci: np.ndarray, clean_hours: list[int]) -> go.Figure:
    hours = list(range(24))
    colors = [GREEN if h in clean_hours else SLATE for h in hours]
    fig = go.Figure(go.Bar(x=hours, y=ci, marker_color=colors))
    fig.update_layout(xaxis_title="Hour of day", yaxis_title="gCO2e per kWh", xaxis=dict(dtick=2))
    return _style(fig, "Illustrative grid intensity by hour (green = cleanest hours)", height=340)


def org_projection(base: pd.DataFrame, opt: pd.DataFrame, col: str, ylabel: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=base["year"], y=base[col], name="Baseline", mode="lines+markers", line=dict(color=RED, width=3))
    fig.add_scatter(x=opt["year"], y=opt[col], name="With levers", mode="lines+markers", line=dict(color=TEAL, width=3))
    fig.update_layout(xaxis_title="Year", yaxis_title=ylabel, xaxis=dict(dtick=1))
    return _style(fig, title)


def waterfall(df: pd.DataFrame, col: str, delta_col: str, title: str, unit: str) -> go.Figure:
    measures = ["absolute"] + ["relative"] * (len(df) - 2) + ["total"]
    y = [df[col].iloc[0]] + list(df[delta_col].iloc[1:-1]) + [df[col].iloc[-1]]
    fig = go.Figure(
        go.Waterfall(
            x=list(df["step"]),
            y=y,
            measure=measures,
            decreasing=dict(marker=dict(color=GREEN)),
            increasing=dict(marker=dict(color=RED)),
            totals=dict(marker=dict(color=TEAL)),
            connector=dict(line=dict(color="#94A3B8")),
        )
    )
    fig.update_layout(yaxis_title=unit, showlegend=False)
    return _style(fig, title)


def distribution_hist(values: np.ndarray, title: str, xlabel: str, color: str) -> go.Figure:
    p5, p50, p95 = np.percentile(values, [5, 50, 95])
    fig = go.Figure(go.Histogram(x=values, nbinsx=60, marker_color=color, opacity=0.85))
    for v, name, dash in [(p5, "P5", "dot"), (p50, "Median", "solid"), (p95, "P95", "dot")]:
        fig.add_vline(x=v, line=dict(color="#0F172A", dash=dash, width=2), annotation_text=name)
    fig.update_layout(xaxis_title=xlabel, yaxis_title="Simulations", showlegend=False)
    return _style(fig, title, height=330)


def tornado(df: pd.DataFrame, title: str) -> go.Figure:
    d = df.copy()
    d["span"] = d["low"].abs() + d["high"].abs()
    d = d.sort_values("span")
    fig = go.Figure()
    fig.add_bar(y=d["parameter"], x=d["low"], orientation="h", name="Low case", marker_color=BLUE)
    fig.add_bar(y=d["parameter"], x=d["high"], orientation="h", name="High case", marker_color=AMBER)
    fig.update_layout(barmode="relative", xaxis_title="% change vs base case")
    return _style(fig, title, height=340)


def score_gauge(score: float) -> go.Figure:
    color = GREEN if score >= 75 else AMBER if score >= 50 else RED
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number=dict(suffix="/100"),
            gauge=dict(
                axis=dict(range=[0, 100]),
                bar=dict(color=color),
                steps=[
                    dict(range=[0, 50], color="#FEE2E2"),
                    dict(range=[50, 75], color="#FEF3C7"),
                    dict(range=[75, 100], color="#DCFCE7"),
                ],
            ),
        )
    )
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0), template="plotly_white")
    return fig
