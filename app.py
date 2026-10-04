"""GenAI Footprint Lab - Streamlit app.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts
from src.calculator import (
    call_footprint,
    equivalents,
    footprint_for_task,
    footprint_from_energy,
    passport_footprint,
    text_energy_wh,
)
from src.data import (
    DEFAULT_REGION,
    INPUT_TOKEN_WEIGHT,
    MODEL_CLASSES,
    REASONING_TOKEN_MULTIPLIER,
    REFERENCES,
    REGIONS,
    TASKS,
)
from src.fallback import demo_advisor, demo_summary
from src.llm import DEMO, PROVIDERS, LLMConfig, LLMError, LLMResult, call_llm
from src.prompt_analyzer import analyse, estimate_tokens, tips
from src.prompts import (
    ADVISOR_SYSTEM,
    JUDGE_SYSTEM,
    OPTIMIZER_SYSTEM,
    SUMMARY_SYSTEM,
    build_context,
    org_to_text,
    parse_judge,
)
from src.report import build_markdown_report
from src.scenarios import (
    Levers,
    OrgParams,
    diurnal_profile,
    lever_waterfall,
    monte_carlo,
    project_org,
    sensitivity,
    summarise_distribution,
    timing_savings,
)
from src.utils import fmt, fmt_co2, fmt_energy, fmt_water

st.set_page_config(
    page_title="GenAI Footprint Lab",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_KEYS = list(MODEL_CLASSES)
model_label = lambda k: MODEL_CLASSES[k]["label"]  # noqa: E731
task_label = lambda k: TASKS[k]["label"]  # noqa: E731


MAX_AI_CALLS = 40  # per browser session, protects API budgets
st.session_state.setdefault("ai_calls", [])
st.session_state.setdefault("chat", [])


def _secret(name: str) -> str:
    """Read an API key from Streamlit secrets (empty string if none configured)."""
    try:
        value = st.secrets.get(name)
        return str(value) if value else ""
    except Exception:  # noqa: BLE001 - no secrets file is a normal situation
        return ""


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🌍 GenAI Footprint Lab")
    st.caption("Energy, carbon and water cost of generative AI")
    region = st.selectbox(
        "Your region / grid",
        list(REGIONS),
        index=list(REGIONS).index(DEFAULT_REGION),
        help="Sets grid carbon intensity, data-centre efficiency and water factors used across the app.",
    )
    r = REGIONS[region]
    st.markdown(
        f"**Grid:** {r['ci']} gCO2e/kWh  \n**PUE:** {r['pue']}  \n"
        f"**On-site water:** {r['wue']} L/kWh  \n**Electricity water:** {r['ewif']} L/kWh"
    )
    st.divider()
    st.markdown("**🤖 GenAI settings**")
    ai_provider = st.selectbox(
        "GenAI provider",
        [DEMO, *PROVIDERS],
        help="Demo mode needs no key and uses rule-based answers. Pick a provider and add a key for live generative AI.",
    )
    ai_cfg: LLMConfig | None = None
    if ai_provider != DEMO:
        pinfo = PROVIDERS[ai_provider]
        secret_key = _secret(pinfo["secret"])
        if secret_key:
            st.caption("🔑 API key loaded from app secrets.")
            api_key = secret_key
        else:
            api_key = st.text_input(
                f"{ai_provider} API key",
                type="password",
                help="Used only for your requests in this session. Never stored or logged by this app.",
            )
        ai_model = st.text_input("Model name", value=pinfo["default_model"], key=f"model_{pinfo['id']}")
        if api_key.strip() and ai_model.strip():
            ai_cfg = LLMConfig(ai_provider, api_key.strip(), ai_model.strip())
        else:
            st.caption("Enter a key to switch from demo to live GenAI.")
    ai_est_model = st.selectbox(
        "Model size to assume for the AI calls' own footprint",
        MODEL_KEYS,
        index=1,
        format_func=model_label,
    )
    st.divider()
    st.caption(
        "All figures are order-of-magnitude estimates from editable assumptions "
        "(`src/data.py`). See the Methodology tab for limitations."
    )

st.title("GenAI Footprint Lab")
st.markdown(
    "##### How much energy, carbon and water does generative AI really use - "
    "and what actually reduces it?"
)

passport_ctx: dict | None = None  # filled by the Passport tab, used by AI Studio
org_ctx: dict | None = None       # filled by the Organisation tab, used by AI Studio

tab_over, tab_ai, tab_calc, tab_prompt, tab_region, tab_org, tab_mc, tab_method = st.tabs(
    [
        "🏠 Overview",
        "🤖 AI Studio",
        "🧾 My Footprint Passport",
        "✂️ Prompt Diet",
        "🌐 Region & Timing",
        "🏢 Organisation Planner",
        "🎲 Uncertainty Lab",
        "📚 Methodology",
    ]
)

# --------------------------------------------------------------------------- #
# 1. Overview
# --------------------------------------------------------------------------- #
with tab_over:
    st.markdown(
        "Every AI answer runs on GPUs in a data centre that draws electricity (often from fossil "
        "fuels) and evaporates water to stay cool. One request is tiny; billions are not. "
        "This lab turns those hidden costs into numbers you can explore, compare and reduce."
    )

    st.subheader(f"One detailed chat answer in {region}")
    cols = st.columns(len(MODEL_KEYS))
    for col, m in zip(cols, MODEL_KEYS):
        fp = footprint_for_task("detailed_chat", m, region)
        with col:
            st.markdown(f"**{model_label(m)}**")
            st.metric("Energy", fmt_energy(fp.facility_wh), border=True)
            st.metric("Carbon", fmt_co2(fp.co2_g), border=True)
            st.metric("Water", fmt_water(fp.water_ml), border=True)

    rows = []
    for t in TASKS:
        for m in MODEL_KEYS:
            fp = footprint_for_task(t, m, region)
            rows.append({"task": task_label(t), "model": model_label(m), "co2_g": fp.co2_g})
    st.plotly_chart(
        charts.task_by_model_bar(
            pd.DataFrame(rows), "co2_g", "g CO2e per request (log scale)",
            f"Carbon per request, by task and model size ({region})",
        ),
        width="stretch",
    )

    chat = footprint_for_task("detailed_chat", "medium", region)
    img = footprint_for_task("image", "medium", region)
    vid = footprint_for_task("video", "medium", region)
    dirty = footprint_for_task("detailed_chat", "medium", "India (national grid)")
    clean = footprint_for_task("detailed_chat", "medium", "Norway (hydro)")
    st.subheader("Key insights (computed live from the model)")
    st.markdown(
        f"- **Not all requests are equal:** on a medium model, one image costs about "
        f"**{fmt(img.co2_g / chat.co2_g)}x** the carbon of a chat answer, and a 5-second video about "
        f"**{fmt(vid.co2_g / chat.co2_g)}x**.\n"
        f"- **Where you run it matters:** the same chat answer emits about "
        f"**{fmt(dirty.co2_g / clean.co2_g)}x** more CO2e on the Indian grid than on Norway's hydro grid.\n"
        f"- **Model size matters:** a small model uses roughly "
        f"**{fmt(footprint_for_task('detailed_chat', 'large', region).co2_g / footprint_for_task('detailed_chat', 'small', region).co2_g)}x** "
        f"less carbon than a frontier model for the same answer.\n"
        f"- **Reasoning modes are expensive:** hidden thinking tokens can multiply the energy of a "
        f"request by up to **{REASONING_TOKEN_MULTIPLIER:.0f}x** in this model."
    )

    st.subheader("How to use this app")
    st.markdown(
        "1. **AI Studio** - generative AI that advises on your results, optimises prompts and writes executive memos.\n"
        "2. **My Footprint Passport** - enter your weekly AI use and get your yearly footprint.\n"
        "3. **Prompt Diet** - paste a prompt and see how much leaner it could be.\n"
        "4. **Region & Timing** - compare grids and test carbon-aware scheduling.\n"
        "5. **Organisation Planner** - project a company's 3-year footprint and test reduction levers.\n"
        "6. **Uncertainty Lab** - see how confident (or not) the estimates are.\n"
        "7. **Methodology** - formulas, assumptions, limitations and references."
    )

# --------------------------------------------------------------------------- #
# 2. Personal footprint passport
# --------------------------------------------------------------------------- #
with tab_calc:
    st.subheader("My annual GenAI footprint")
    st.caption("Enter how many of each activity you do per week.")

    defaults = {
        "quick_qa": 20, "detailed_chat": 15, "code_gen": 5, "summarise_doc": 3,
        "draft_report": 1, "translation": 1, "rag_answer": 2, "image": 3, "video": 0,
    }
    c_inputs = st.columns(3)
    weekly: dict[str, float] = {}
    for i, (t, d) in enumerate(defaults.items()):
        with c_inputs[i % 3]:
            weekly[t] = st.number_input(task_label(t), min_value=0, max_value=5000, value=d, step=1, key=f"w_{t}")

    c1, c2 = st.columns(2)
    with c1:
        model = st.selectbox("Model size you mostly use", MODEL_KEYS, index=1, format_func=model_label, key="pp_model")
    with c2:
        reasoning_share = st.slider(
            "Share of text requests using 'thinking / reasoning' mode", 0, 100, 10, key="pp_reason"
        ) / 100.0

    total, rows = passport_footprint(weekly, model, region, reasoning_share)

    if not rows:
        st.info("Enter at least one activity to see your footprint.")
    else:
        breakdown = pd.DataFrame(rows).round(3)
        m1, m2, m3 = st.columns(3)
        m1.metric("Energy per year", fmt_energy(total.facility_wh), border=True)
        m2.metric("Carbon per year", fmt_co2(total.co2_g), border=True)
        m3.metric("Water per year", fmt_water(total.water_ml), border=True)

        st.markdown("**In everyday terms**")
        eq = equivalents(total)
        eq_cols = st.columns(6)
        for col, e in zip(eq_cols, eq):
            col.metric(f"{e['icon']} {e['label']}", fmt(e["value"]))

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(
                charts.breakdown_donut(
                    list(breakdown["Activity"]), list(breakdown["CO2e (kg/yr)"]), "Where your carbon comes from"
                ),
                width="stretch",
            )
        with right:
            st.dataframe(breakdown, width="stretch", hide_index=True)

        # what-if insights (all computed)
        insights: list[str] = []
        t_small, _ = passport_footprint(weekly, "small", region, reasoning_share)
        if model != "small" and total.co2_g > 0:
            insights.append(
                f"Using small models where they are good enough would cut your carbon by "
                f"{(1 - t_small.co2_g / total.co2_g) * 100:.0f}%."
            )
        if reasoning_share > 0:
            t_nr, _ = passport_footprint(weekly, model, region, 0.0)
            insights.append(
                f"Using reasoning mode only when needed (instead of {reasoning_share:.0%} of requests) "
                f"would cut carbon by {(1 - t_nr.co2_g / total.co2_g) * 100:.0f}%."
            )
        greenest = min(REGIONS, key=lambda k: passport_footprint(weekly, model, k, reasoning_share)[0].co2_g)
        t_g, _ = passport_footprint(weekly, model, greenest, reasoning_share)
        if greenest != region and total.co2_g > 0:
            insights.append(
                f"Running the same workload in {greenest} would cut carbon by "
                f"{(1 - t_g.co2_g / total.co2_g) * 100:.0f}%."
            )
        top = breakdown.sort_values("CO2e (kg/yr)", ascending=False).iloc[0]
        insights.append(
            f"Your biggest carbon source is '{top['Activity']}' "
            f"({top['CO2e (kg/yr)'] / breakdown['CO2e (kg/yr)'].sum() * 100:.0f}% of the total)."
        )

        st.markdown("**What would reduce it**")
        for line in insights:
            st.markdown(f"- {line}")

        passport_ctx = {
            "energy_wh": total.facility_wh,
            "co2_g": total.co2_g,
            "water_ml": total.water_ml,
            "model_label": model_label(model),
            "reasoning_share": reasoning_share,
            "breakdown": breakdown.to_dict("records"),
            "insights": insights,
        }
        report_md = build_markdown_report(total, breakdown, region, model_label(model), reasoning_share, insights)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download report (.md)", report_md, "genai_footprint_passport.md", "text/markdown")
        d2.download_button("⬇️ Download breakdown (.csv)", breakdown.to_csv(index=False), "breakdown.csv", "text/csv")

# --------------------------------------------------------------------------- #
# 3. Prompt Diet
# --------------------------------------------------------------------------- #
SAMPLE_PROMPT = (
    "Hi! I was wondering if you could please help me. Could you please write a very short summary "
    "of the following text in order to help me prepare for my exam? Please make it really simple. "
    "Please make it really simple. If possible, I would like you to use bullet points. "
    "Thank you so much in advance!"
)

with tab_prompt:
    st.subheader("Prompt Diet - how lean is your prompt?")
    st.caption("A rule-based analyser (no API key, nothing leaves your browser session).")
    prompt = st.text_area("Paste a prompt", SAMPLE_PROMPT, height=150, key="pd_prompt")

    p1, p2, p3, p4 = st.columns(4)
    pd_model = p1.selectbox("Model size", MODEL_KEYS, index=1, format_func=model_label, key="pd_model")
    out_tokens = p2.number_input("Typical answer length (tokens)", 20, 8000, 600, step=50, key="pd_out")
    concise = p3.slider("Shorter answer when you ask for concise output (%)", 0, 60, 25, key="pd_concise") / 100.0
    calls_per_day = p4.number_input("Times you send prompts like this per day", 1, 1_000_000, 50, key="pd_calls")

    if not prompt.strip():
        st.info("Paste a prompt to analyse it.")
    else:
        res = analyse(prompt)
        g_col, m_col = st.columns([1, 2])
        with g_col:
            st.plotly_chart(charts.score_gauge(res["score"]), width="stretch")
            st.caption("Prompt efficiency score")
        with m_col:
            a, b, c = st.columns(3)
            a.metric("Tokens (est.)", res["tokens"], border=True)
            b.metric("After diet", res["lean_tokens"], delta=f"-{res['saved_tokens']}", delta_color="inverse", border=True)
            c.metric("Filler phrases", res["filler_total"], border=True)
            if res["filler_hits"]:
                st.markdown(
                    "Found: " + ", ".join(f"`{k}` x{v}" for k, v in res["filler_hits"].items())
                    + (f"; **{res['repeated_sentences']}** repeated sentence(s)" if res["repeated_sentences"] else "")
                )

        st.markdown("**Leaner rewrite** (automatic, review before using)")
        st.code(res["lean_text"] or "(empty)", language=None)

        base_wh = text_energy_wh(pd_model, res["tokens"], out_tokens)
        trim_wh = text_energy_wh(pd_model, res["lean_tokens"], out_tokens)
        both_wh = text_energy_wh(pd_model, res["lean_tokens"], out_tokens * (1 - concise))
        n_year = calls_per_day * 365
        f_base = footprint_from_energy(base_wh * n_year, region)
        f_both = footprint_from_energy(both_wh * n_year, region)
        saved_co2 = f_base.co2_g - f_both.co2_g
        saved_water = f_base.water_ml - f_both.water_ml

        st.markdown("**Impact over a year**")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Energy saved", fmt_energy(f_base.facility_wh - f_both.facility_wh), border=True)
        s2.metric("CO2e saved", fmt_co2(saved_co2), border=True)
        s3.metric("Water saved", fmt_water(saved_water), border=True)
        from_prompt = (base_wh - trim_wh) / base_wh * 100 if base_wh else 0
        from_answer = (trim_wh - both_wh) / base_wh * 100 if base_wh else 0
        s4.metric("Share of saving from the answer", f"{from_answer / max(from_prompt + from_answer, 1e-9) * 100:.0f}%", border=True)
        st.caption(
            f"Trimming the prompt saves {from_prompt:.1f}% of per-request energy; asking for a shorter answer "
            f"saves {from_answer:.1f}%. Reading tokens is weighted at {INPUT_TOKEN_WEIGHT:.0%} of generating them, "
            "so output length is the bigger lever."
        )
        st.markdown("**Tips**")
        for tip in tips(res):
            st.markdown(f"- {tip}")

# --------------------------------------------------------------------------- #
# 4. Region & timing
# --------------------------------------------------------------------------- #
with tab_region:
    st.subheader("Where and when you run AI changes its footprint")
    r1, r2, r3 = st.columns(3)
    rg_task = r1.selectbox("Workload", list(TASKS), index=1, format_func=task_label, key="rg_task")
    rg_model = r2.selectbox("Model size", MODEL_KEYS, index=1, format_func=model_label, key="rg_model")
    rg_n = r3.number_input("Number of requests", 1, 100_000_000, 1000, step=1000, key="rg_n")

    rr = []
    for k in REGIONS:
        fp = footprint_for_task(rg_task, rg_model, k, rg_n)
        rr.append({"region": k, "co2_g": fp.co2_g, "water_ml": fp.water_ml, "energy_wh": fp.facility_wh})
    rdf = pd.DataFrame(rr)

    cA, cB = st.columns(2)
    with cA:
        st.plotly_chart(
            charts.region_bars(rdf.assign(co2_kg=rdf["co2_g"] / 1000), "co2_kg", "kg CO2e", "Carbon by region", charts.TEAL),
            width="stretch",
        )
    with cB:
        st.plotly_chart(
            charts.region_bars(rdf.assign(water_l=rdf["water_ml"] / 1000), "water_l", "litres", "Water by region", charts.BLUE),
            width="stretch",
        )
    st.plotly_chart(charts.carbon_water_scatter(rdf), width="stretch")
    best_c = rdf.loc[rdf["co2_g"].idxmin(), "region"]
    best_w = rdf.loc[rdf["water_ml"].idxmin(), "region"]
    if best_c != best_w:
        st.info(f"Trade-off: lowest carbon is **{best_c}**, but lowest water is **{best_w}**. Optimising one metric can worsen the other.")
    else:
        st.info(f"**{best_c}** is lowest on both carbon and water for this workload.")

    st.divider()
    st.subheader("Carbon-aware timing")
    st.caption("Uses a synthetic, illustrative 24-hour grid curve (evening peak, midday solar dip) - not live grid data.")
    t1, t2, t3, t4 = st.columns(4)
    solar = t1.slider("Midday solar dip (%)", 0, 60, 25, key="tm_solar") / 100
    peak = t2.slider("Evening demand peak (%)", 0, 40, 20, key="tm_peak") / 100
    flex = t3.slider("Share of workload that can wait (%)", 0, 100, 40, key="tm_flex") / 100
    hrs = t4.slider("Run in the N cleanest hours", 1, 12, 4, key="tm_hrs")
    curve = diurnal_profile(REGIONS[region]["ci"], solar, peak)
    ts = timing_savings(curve, flex, hrs)
    st.plotly_chart(charts.diurnal_chart(curve, ts["clean_hours"]), width="stretch")
    st.success(
        f"Shifting {flex:.0%} of a flexible workload (batch summaries, embeddings, nightly jobs) to hours "
        f"{', '.join(f'{h}:00' for h in ts['clean_hours'])} cuts its carbon by about **{ts['saving_pct']:.1f}%** in {region}."
    )

# --------------------------------------------------------------------------- #
# 5. Organisation planner
# --------------------------------------------------------------------------- #
with tab_org:
    st.subheader("Organisation planner - 3-year footprint and reduction levers")

    o1, o2 = st.columns(2)
    with o1:
        st.markdown("**Usage**")
        employees = st.number_input("Employees", 10, 1_000_000, 5000, step=100, key="o_emp")
        adoption = st.slider("Employees using GenAI (%)", 1, 100, 60, key="o_adopt") / 100
        qpd = st.slider("Text requests per user per workday", 1, 200, 20, key="o_qpd")
        ipd = st.slider("Images per user per workday", 0.0, 20.0, 0.5, 0.5, key="o_ipd")
        growth = st.slider("Yearly usage growth (%)", 0, 200, 35, key="o_growth") / 100
        years = st.slider("Years to project", 1, 6, 3, key="o_years")
        s_small = st.slider("Traffic on small models (%)", 0, 100, 20, key="o_small")
        s_large = st.slider("Traffic on large / frontier models (%)", 0, 100, 50, key="o_large")
        o_reason = st.slider("Requests using reasoning mode (%)", 0, 100, 10, key="o_reason") / 100
    with o2:
        st.markdown("**Reduction levers**")
        cache = st.slider("Caching / de-duplication (% requests avoided)", 0, 60, 15, key="o_cache") / 100
        trim = st.slider("Token discipline (% fewer tokens)", 0, 60, 20, key="o_trim") / 100
        rightsize = st.slider("Right-sizing (% of large traffic moved to small models)", 0, 100, 40, key="o_right") / 100
        tgt_options = [k for k in REGIONS if k != region]
        target = st.selectbox("Move some workload to", tgt_options, index=tgt_options.index("Norway (hydro)") if "Norway (hydro)" in tgt_options else 0, key="o_target")
        shift = st.slider("Share of workload moved (%)", 0, 100, 30, key="o_shift") / 100

    if s_small + s_large > 100:
        st.error("Small + large traffic shares exceed 100%. Reduce one of them.")
    else:
        shares = {"small": s_small / 100, "medium": (100 - s_small - s_large) / 100, "large": s_large / 100}
        params = OrgParams(
            employees=int(employees), adoption=adoption, queries_per_day=float(qpd), images_per_day=float(ipd),
            growth=growth, years=int(years), region=region, shares=shares, reasoning_share=o_reason,
        )
        lv = Levers(cache, trim, rightsize, target, shift)
        base_df = project_org(params, Levers(region_target=target))
        opt_df = project_org(params, lv)
        wf = lever_waterfall(params, lv)

        b_co2, o_co2 = base_df["co2_t"].sum(), opt_df["co2_t"].sum()
        b_w, o_w = base_df["water_m3"].sum(), opt_df["water_m3"].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Baseline CO2e ({years} yr)", f"{fmt(b_co2)} t", border=True)
        k2.metric("With levers", f"{fmt(o_co2)} t", delta=f"-{(1 - o_co2 / b_co2) * 100:.0f}%", delta_color="inverse", border=True)
        k3.metric(f"Baseline water ({years} yr)", f"{fmt(b_w)} m3", border=True)
        k4.metric("With levers", f"{fmt(o_w)} m3", delta=f"-{(1 - o_w / b_w) * 100:.0f}%", delta_color="inverse", border=True)

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(charts.org_projection(base_df, opt_df, "co2_t", "tonnes CO2e", "Yearly carbon"), width="stretch")
        with g2:
            st.plotly_chart(charts.waterfall(wf, "co2_t", "co2_delta", f"Which lever saves what ({years}-year CO2e)", "tonnes CO2e"), width="stretch")
        st.plotly_chart(charts.org_projection(base_df, opt_df, "water_m3", "cubic metres", "Yearly water"), width="stretch")

        org_ctx = {
            "employees": int(employees), "adoption": adoption, "years": int(years), "region": region,
            "base_co2": b_co2, "opt_co2": o_co2, "base_water": b_w, "opt_water": o_w,
            "co2_cut": (1 - o_co2 / b_co2) * 100 if b_co2 else 0.0,
            "water_cut": (1 - o_w / b_w) * 100 if b_w else 0.0,
            "cache": cache, "trim": trim, "rightsize": rightsize, "shift": shift, "target": target,
            "waterfall": wf.to_dict("records"),
        }
        st.markdown("**Lever impact table**")
        show = wf.rename(columns={"step": "Step", "co2_t": "CO2e (t)", "water_m3": "Water (m3)",
                                  "co2_delta": "CO2e change (t)", "water_delta": "Water change (m3)"}).round(2)
        st.dataframe(show, width="stretch", hide_index=True)
        st.download_button("⬇️ Download projection (.csv)", opt_df.round(3).to_csv(index=False), "org_projection.csv", "text/csv")

# --------------------------------------------------------------------------- #
# 6. Uncertainty lab
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _cached_mc(task, model, reg, n, draws, reasoning):
    return monte_carlo(task, model, reg, n_queries=n, draws=draws, reasoning=reasoning)


with tab_mc:
    st.subheader("Uncertainty lab - how sure are these numbers?")
    st.caption("Monte Carlo simulation samples energy, PUE, grid intensity and water factors from plausible ranges.")
    u1, u2, u3, u4, u5 = st.columns(5)
    mc_task = u1.selectbox("Workload", list(TASKS), index=1, format_func=task_label, key="mc_task")
    mc_model = u2.selectbox("Model size", MODEL_KEYS, index=1, format_func=model_label, key="mc_model")
    mc_n = u3.number_input("Requests", 1, 10_000_000, 1000, step=500, key="mc_n")
    mc_draws = u4.selectbox("Simulations", [2000, 5000, 10000], index=1, key="mc_draws")
    mc_reason = u5.checkbox("Reasoning mode", False, key="mc_reason")

    mc = _cached_mc(mc_task, mc_model, region, int(mc_n), int(mc_draws), mc_reason)
    summary = summarise_distribution(mc)

    h1, h2 = st.columns(2)
    with h1:
        st.plotly_chart(charts.distribution_hist(mc["co2_kg"].to_numpy(), f"Carbon for {int(mc_n):,} requests", "kg CO2e", charts.TEAL), width="stretch")
    with h2:
        st.plotly_chart(charts.distribution_hist(mc["water_l"].to_numpy(), f"Water for {int(mc_n):,} requests", "litres", charts.BLUE), width="stretch")

    st.markdown("**Distribution summary**")
    st.dataframe(summary.round(4), width="stretch", hide_index=True)
    spread = summary.loc[summary["metric"] == "co2_kg", "P95 / P5"].iloc[0]
    st.info(
        f"The 95th-percentile carbon estimate is about **{fmt(spread)}x** the 5th-percentile one. "
        "Treat single-number footprints as ranges, and compare options on the same assumptions."
    )

    sens = sensitivity(mc_task, mc_model, region, mc_reason)
    t1c, t2c = st.columns(2)
    with t1c:
        st.plotly_chart(charts.tornado(sens["co2"], "What drives carbon uncertainty"), width="stretch")
    with t2c:
        st.plotly_chart(charts.tornado(sens["water"], "What drives water uncertainty"), width="stretch")

# --------------------------------------------------------------------------- #
# 8. AI Studio (generative AI integration)
# --------------------------------------------------------------------------- #
def run_ai(feature: str, system: str, messages: list[dict], max_tokens: int = 700) -> tuple[LLMResult, float]:
    """Call the configured LLM, log its own footprint, return (result, IT-energy Wh of the call)."""
    assert ai_cfg is not None
    if len(st.session_state["ai_calls"]) >= MAX_AI_CALLS:
        raise LLMError(f"Session limit of {MAX_AI_CALLS} AI calls reached. Refresh the page to start a new session.")
    cfg = LLMConfig(ai_cfg.provider, ai_cfg.api_key, ai_cfg.model, max_tokens=max_tokens, temperature=ai_cfg.temperature)
    with st.spinner("Asking the model..."):
        res = call_llm(cfg, system, messages)
    fp = call_footprint(res.input_tokens, res.output_tokens, ai_est_model, region)
    st.session_state["ai_calls"].append(
        {
            "Feature": feature,
            "Provider": cfg.provider,
            "Model": cfg.model,
            "Input tokens": res.input_tokens,
            "Output tokens": res.output_tokens,
            "Energy (Wh)": fp.facility_wh,
            "CO2e (g)": fp.co2_g,
            "Water (mL)": fp.water_ml,
        }
    )
    return res, text_energy_wh(ai_est_model, res.input_tokens, res.output_tokens)


with tab_ai:
    st.subheader("🤖 AI Studio - generative AI working on your footprint data")
    if ai_cfg:
        st.success(f"Live GenAI: **{ai_cfg.provider}** - model `{ai_cfg.model}`")
    else:
        st.info(
            "**Demo mode** - features run on rule-based logic so everything works without a key. "
            "Choose a provider and paste an API key in the sidebar to switch on live generative AI."
        )

    sub_chat, sub_opt, sub_sum, sub_self = st.tabs(
        ["💬 Footprint Advisor", "🪄 Smart Prompt Optimizer", "📝 Executive Memo", "📡 The AI's own footprint"]
    )

    # ---- Advisor (grounded chat) ----
    with sub_chat:
        st.caption(
            "Ask about your results. The model receives this app's assumptions and your computed numbers as "
            "context (grounding), so answers come from data, not guesses."
        )
        ctx = build_context(region, passport_ctx, org_ctx)
        suggestions = [
            "What is the single biggest way for me to cut my footprint?",
            "Why can a low-carbon grid still use a lot of water?",
            "Is it worth switching to a smaller model?",
        ]
        s_cols = st.columns(len(suggestions))
        for i, (col, text) in enumerate(zip(s_cols, suggestions)):
            if col.button(text, key=f"sugg_{i}"):
                st.session_state["pending_q"] = text

        for m in st.session_state["chat"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        question = st.chat_input("Ask the Footprint Advisor...", key="ai_chat_input")
        if not question:
            question = st.session_state.pop("pending_q", None)
        if question:
            st.session_state["chat"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    if ai_cfg:
                        history = list(st.session_state["chat"][-8:])
                        while history and history[0]["role"] != "user":
                            history.pop(0)
                        result, _ = run_ai("Advisor", ADVISOR_SYSTEM.format(context=ctx), history)
                        answer = result.text
                    else:
                        answer = demo_advisor(question, region, passport_ctx, org_ctx)
                except LLMError as exc:
                    answer = f"⚠️ {exc}"
                st.markdown(answer)
            st.session_state["chat"].append({"role": "assistant", "content": answer})
        if st.session_state["chat"] and st.button("Clear conversation", key="clear_chat"):
            st.session_state["chat"] = []
            st.rerun()
        with st.expander("What the model is given (grounding context)"):
            st.code(ctx, language=None)

    # ---- Smart prompt optimizer ----
    with sub_opt:
        st.caption(
            "An LLM rewrites your prompt to be shorter without losing requirements; a second LLM call (the judge) "
            "checks the meaning was preserved. Then we measure the footprint saved and the break-even point."
        )
        opt_prompt = st.text_area("Prompt to optimise", SAMPLE_PROMPT, height=130, key="ai_opt_prompt")
        oc1, oc2, oc3 = st.columns(3)
        opt_model = oc1.selectbox("Model that will run the prompt", MODEL_KEYS, index=1, format_func=model_label, key="ai_opt_model")
        opt_out = oc2.number_input("Typical answer length (tokens)", 20, 8000, 600, step=50, key="ai_opt_out")
        use_judge = oc3.checkbox("Verify meaning with an LLM judge", value=True, key="ai_opt_judge", disabled=ai_cfg is None)

        if st.button("🪄 Optimise prompt", key="ai_opt_btn"):
            if not opt_prompt.strip():
                st.warning("Paste a prompt first.")
            else:
                rule = analyse(opt_prompt)
                try:
                    judge, call_wh, rewritten = None, 0.0, rule["lean_text"]
                    if ai_cfg:
                        res_opt, call_wh = run_ai(
                            "Optimizer", OPTIMIZER_SYSTEM,
                            [{"role": "user", "content": f"<prompt>\n{opt_prompt}\n</prompt>"}], max_tokens=800,
                        )
                        rewritten = res_opt.text
                        if use_judge:
                            res_j, wh_j = run_ai(
                                "Judge", JUDGE_SYSTEM,
                                [{"role": "user", "content":
                                  f"ORIGINAL:\n<prompt>\n{opt_prompt}\n</prompt>\n\nREWRITTEN:\n<prompt>\n{rewritten}\n</prompt>"}],
                                max_tokens=150,
                            )
                            judge = parse_judge(res_j.text)
                            call_wh += wh_j
                    st.session_state["opt_result"] = {
                        "original": opt_prompt, "rewritten": rewritten, "rule_lean": rule["lean_text"],
                        "judge": judge, "call_wh": call_wh, "live": ai_cfg is not None,
                    }
                except LLMError as exc:
                    st.error(str(exc))

        opt_res = st.session_state.get("opt_result")
        if opt_res:
            t_orig = estimate_tokens(opt_res["original"])
            t_ai = estimate_tokens(opt_res["rewritten"])
            t_rule = estimate_tokens(opt_res["rule_lean"])
            label = "AI rewrite" if opt_res["live"] else "Rule-based rewrite (demo)"
            a, b, c = st.columns(3)
            a.metric("Original tokens (est.)", t_orig, border=True)
            b.metric(f"{label} tokens", t_ai, delta=f"{t_ai - t_orig}", delta_color="inverse", border=True)
            c.metric("Rule-based tokens", t_rule, delta=f"{t_rule - t_orig}", delta_color="inverse", border=True)

            left, right = st.columns(2)
            left.markdown("**Original**")
            left.code(opt_res["original"], language=None)
            right.markdown(f"**{label}**")
            right.code(opt_res["rewritten"], language=None)

            if opt_res["judge"]:
                score, reason = opt_res["judge"]
                st.metric("Meaning preserved (LLM judge)", f"{score}/5" if score else "n/a", border=True)
                st.caption(f"Judge's reason: {reason}")
                if score is not None and score <= 3:
                    st.warning("The judge thinks requirements were lost - review the rewrite before using it.")

            saving_wh = text_energy_wh(opt_model, t_orig, opt_out) - text_energy_wh(opt_model, t_ai, opt_out)
            f1000 = footprint_from_energy(max(saving_wh, 0) * 1000, region)
            st.markdown("**Footprint impact**")
            i1, i2 = st.columns(2)
            i1.metric("CO2e saved per 1,000 uses", fmt_co2(f1000.co2_g), border=True)
            i2.metric("Water saved per 1,000 uses", fmt_water(f1000.water_ml), border=True)
            if opt_res["live"] and saving_wh > 0 and opt_res["call_wh"] > 0:
                st.info(
                    f"Optimising cost about {fmt(opt_res['call_wh'])} Wh of IT energy. This prompt saves "
                    f"{fmt(saving_wh)} Wh per use, so it **pays back after ~{fmt(opt_res['call_wh'] / saving_wh)} uses**. "
                    "Optimise prompts you will reuse often (templates, system prompts, automations)."
                )
            elif opt_res["live"]:
                st.info("This rewrite does not reduce tokens, so there is no energy payback.")

    # ---- Executive memo ----
    with sub_sum:
        st.caption("Turns the Organisation Planner results into a short memo for non-technical decision-makers.")
        if not org_ctx:
            st.info("Configure the Organisation Planner tab first (valid traffic shares), then come back.")
        else:
            with st.expander("Data sent to the model"):
                st.code(org_to_text(org_ctx), language=None)
            if st.button("📝 Generate executive memo", key="ai_memo_btn"):
                try:
                    if ai_cfg:
                        res_m, _ = run_ai(
                            "Executive memo", SUMMARY_SYSTEM,
                            [{"role": "user", "content": "DATA:\n" + org_to_text(org_ctx)}], max_tokens=500,
                        )
                        st.session_state["memo"] = res_m.text
                    else:
                        st.session_state["memo"] = demo_summary(org_ctx)
                except LLMError as exc:
                    st.error(str(exc))
            if st.session_state.get("memo"):
                st.markdown(st.session_state["memo"])
                st.download_button("⬇️ Download memo (.md)", st.session_state["memo"], "executive_memo.md", "text/markdown")
                st.caption("If you change the planner settings, generate the memo again.")

    # ---- Self-footprint ----
    with sub_self:
        st.caption(
            "This app practises what it measures: every live AI call is logged and converted to energy, carbon "
            "and water with the same model used elsewhere in the lab."
        )
        calls = st.session_state["ai_calls"]
        if not calls:
            st.info("No live AI calls yet. In demo mode nothing is sent to a model.")
        else:
            cdf = pd.DataFrame(calls)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("AI calls this session", len(cdf), border=True)
            e2.metric("Energy", fmt_energy(cdf["Energy (Wh)"].sum()), border=True)
            e3.metric("Carbon", fmt_co2(cdf["CO2e (g)"].sum()), border=True)
            e4.metric("Water", fmt_water(cdf["Water (mL)"].sum()), border=True)
            st.dataframe(cdf.round(4), width="stretch", hide_index=True)
            st.caption(
                f"Assumes the provider's model behaves like a {model_label(ai_est_model)} model in {region}. "
                "Token counts come from the provider's own usage report."
            )

# --------------------------------------------------------------------------- #
# 7. Methodology
# --------------------------------------------------------------------------- #
with tab_method:
    st.subheader("Methodology")
    st.markdown("**Step 1 - energy of a request (IT equipment):**")
    st.latex(r"E_{IT} = \frac{w_{model}}{1000}\left(T_{out}\cdot k_{reason} + \alpha\,T_{in}\right)")
    st.markdown("**Step 2 - add data-centre overhead, then convert to carbon and water:**")
    st.latex(r"E_{fac} = E_{IT}\cdot PUE \qquad CO_2e = E_{fac}\cdot CI \qquad W = E_{IT}\cdot WUE + E_{fac}\cdot EWIF")
    st.markdown(
        f"- `w_model`: Wh per 1,000 generated tokens for the model class (editable)\n"
        f"- `alpha` = {INPUT_TOKEN_WEIGHT}: reading a token costs less than generating one\n"
        f"- `k_reason` = {REASONING_TOKEN_MULTIPLIER:.0f} when reasoning mode is on, else 1\n"
        "- `PUE`: facility energy / IT energy; `CI`: grid carbon intensity; `WUE`: on-site cooling water; "
        "`EWIF`: water consumed to generate the electricity"
    )

    st.markdown("**Model-class assumptions**")
    st.dataframe(
        pd.DataFrame(
            [{"Class": v["label"], "Text Wh / 1k tokens": v["text_wh_per_1k_tokens"],
              "Image Wh": v["image_wh"], "Video Wh (5 s)": v["video_wh"]} for v in MODEL_CLASSES.values()]
        ),
        width="stretch", hide_index=True,
    )
    st.markdown("**Region assumptions**")
    st.dataframe(
        pd.DataFrame(
            [{"Region": k, "CI (gCO2e/kWh)": v["ci"], "PUE": v["pue"], "WUE (L/kWh)": v["wue"], "EWIF (L/kWh)": v["ewif"]}
             for k, v in REGIONS.items()]
        ),
        width="stretch", hide_index=True,
    )
    st.markdown("**Task catalogue**")
    st.dataframe(
        pd.DataFrame(
            [{"Task": v["label"], "Kind": v["kind"], "Input tokens": str(v.get("in", "-")), "Output tokens": str(v.get("out", "-"))}
             for v in TASKS.values()]
        ),
        width="stretch", hide_index=True,
    )

    st.markdown("**Generative-AI integration**")
    st.markdown(
        "- **Grounding (context injection):** the Advisor's system prompt contains this app's assumptions and your "
        "computed results, so it answers from data and is told not to invent numbers.\n"
        "- **Prompt optimiser + LLM-as-judge:** one call rewrites the prompt; a second call scores meaning "
        "preservation (1-5). Prompts are wrapped in tags and treated as data to resist prompt injection.\n"
        "- **Executive memo:** structured numbers go in, a plain-language memo comes out.\n"
        "- **Self-measurement:** token usage reported by the provider is converted into the AI calls' own footprint.\n"
        "- **Safeguards:** demo fallback without a key, input length limit, per-session call cap, keys never stored.\n"
        "- **Limits:** LLM output can be wrong or incomplete; always check numbers against the dashboards."
    )

    st.markdown("**Limitations (read before quoting any number)**")
    st.markdown(
        "- Values are *order-of-magnitude assumptions*, not measurements of any specific product.\n"
        "- Real energy depends on hardware generation, batching, quantisation, context length and utilisation.\n"
        "- Training and hardware manufacturing (embodied carbon) are excluded - this models inference only.\n"
        "- Grid intensity is an annual average; the timing chart uses a synthetic curve, not live data.\n"
        "- Water factors vary widely by cooling design and local scarcity; litres are not equally harmful everywhere.\n"
        "- Token counts use a simple heuristic, not a real tokeniser."
    )
    st.markdown("**Ethics note**")
    st.markdown(
        "The goal is informed choices, not guilt. AI can also *reduce* emissions elsewhere; this lab only "
        "quantifies the direct operating footprint so that efficiency decisions can be compared."
    )
    st.markdown("**References that informed the assumptions** (verify before citing)")
    for ref in REFERENCES:
        st.markdown(f"- {ref}")
