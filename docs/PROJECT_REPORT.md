# GenAI Footprint Lab - Project Report

**Module:** Artificial Intelligence & Generative AI (Capstone)
**Deliverable:** Streamlit web application + source code (GitHub)

> Figures in section 6 are outputs of the app's *default assumptions*. They are estimates, not measurements.

---

## 1. Abstract
Generative AI is used billions of times a day, yet the electricity, carbon and water behind each answer are invisible to users and decision-makers. GenAI Footprint Lab is an interactive decision-support tool that estimates the operating footprint of AI requests, shows how much that footprint varies with model size, task type, region and time of day, quantifies the uncertainty in those estimates, and tests which reduction strategies matter most for individuals and organisations.

## 2. Problem statement
Responsible adoption of generative AI requires knowing its resource cost, but public figures are scattered, inconsistent and rarely actionable. Users cannot answer: *How much does my AI use cost the planet? Which of my habits matters most? What should my organisation change first?*

## 3. Objectives
1. Build a transparent, editable model converting AI workload into energy, carbon and water.
2. Compare model sizes, task types (text, image, video, reasoning) and regions, including India.
3. Reveal the carbon-versus-water trade-off between grids.
4. Quantify estimation uncertainty with Monte Carlo and sensitivity analysis.
5. Evaluate four reduction levers for an organisation over three years.
6. Integrate generative AI so an LLM advises on the results, optimises prompts and drafts executive summaries, while the app measures the footprint of its own AI calls.
7. Deploy the tool as a free, public Streamlit app.

## 4. Methodology
**Energy:** `E_IT = (w_model/1000) × (T_out × k_reason + α × T_in)` - energy depends on model class, generated tokens, a lower weight for input tokens (α = 0.15) and a multiplier for hidden reasoning tokens (k = 8). Image and video use per-item energy.

**Facility overhead:** `E_fac = E_IT × PUE`.

**Carbon:** `CO2e = E_fac × CI` (grid carbon intensity, gCO2e/kWh).

**Water:** `W = E_IT × WUE + E_fac × EWIF` - on-site cooling water plus the water consumed generating the electricity.

**Uncertainty:** 5,000 Monte Carlo draws - energy per query (log-normal, σ = 0.5), PUE, grid intensity, WUE and EWIF (triangular). One-at-a-time sensitivity produces tornado charts.

**Organisation planner:** workload = employees × adoption × requests/day × workdays, grown yearly. Levers applied in sequence: caching → token discipline → right-sizing → regional shift; the waterfall attributes savings to each step.

**Prompt Diet:** rule-based analyser: token estimate, regex filler-phrase detection, repeated-sentence detection, efficiency score (0-100) and a leaner rewrite; savings split into "shorter prompt" vs "shorter answer".

**Carbon-aware timing:** synthetic 24-hour intensity curve (evening peak, midday solar dip); shifts a flexible share of work to the N cleanest hours.

### 4.1 Generative-AI integration (AI Studio)
The app is connected to a large language model through a provider-agnostic client (`src/llm.py`) supporting Anthropic Claude, OpenAI and Google Gemini over plain HTTPS. Four GenAI features use it:

1. **Footprint Advisor (grounded chat).** The system prompt is assembled from the app's own assumptions and the user's computed results (`src/prompts.py: build_context`). This is *context injection / grounding*: the model is instructed to answer only from supplied numbers, flag uncertainty and decline off-topic requests, which reduces hallucination.
2. **Smart Prompt Optimizer with LLM-as-judge.** One call rewrites a prompt to be shorter while preserving every requirement; a second call scores meaning preservation from 1 to 5. The app then computes the carbon and water saved per 1,000 uses and the **break-even number of uses** (the optimisation call itself costs energy).
3. **Executive Memo.** Structured planner results go in; a short plain-language memo for non-technical leaders comes out.
4. **Self-measurement.** Every live call's token usage, as reported by the provider, is converted into energy, carbon and water with the same model as the rest of the lab.

**Safeguards:** a rule-based Demo mode so the app works without any key (`src/fallback.py`); input-length limit; per-session call cap; keys supplied per session or via Streamlit secrets and never stored or logged; the Gemini key is sent in a header, not the URL; user text is wrapped in tags and treated as data to resist prompt injection; friendly error handling for bad keys, quotas and network faults.

## 5. System design
- **UI layer:** `app.py` (Streamlit, 8 tabs plus 4 AI sub-tabs, no business logic).
- **Domain layer:** `src/calculator.py`, `src/scenarios.py`, `src/prompt_analyzer.py`.
- **GenAI layer:** `src/llm.py` (provider client), `src/prompts.py` (prompt templates, grounding, judge parser), `src/fallback.py` (demo mode).
- **Data layer:** `src/data.py` (single source of all assumptions).
- **Presentation helpers:** `src/charts.py`, `src/report.py`, `src/utils.py`.
- **Quality:** 32 automated tests - formula and monotonicity unit tests, mocked API tests for all three providers (success, HTTP errors, key-leak checks), and app tests that drive the AI Studio in both demo and mocked live mode.

## 6. Results (default assumptions)
*One detailed chat answer, medium model:*

| Region | Energy (Wh) | CO2e (g) | Water (mL) |
|---|---|---|---|
| India (national grid) | 0.24 | 0.172 | 1.14 |
| France (nuclear-heavy) | 0.21 | 0.012 | 0.59 |
| Brazil (hydro-heavy) | 0.23 | 0.023 | 1.29 |
| Norway (hydro) | 0.19 | 0.006 | 0.20 |

- A medium-model image costs ~**18×** the carbon of a chat answer; a 5-second video ~**2,500×**.
- A frontier model emits ~**10×** the carbon of a small model for the same answer.
- The same answer emits ~**31×** more CO2e on the Indian grid than on Norway's grid.
- Brazil's low-carbon hydro grid has *higher* water than India in this model (large reservoir evaporation factor) - a clear carbon-water trade-off.

*Organisation example (5,000 employees, 60% adoption, 20 requests/day, 35% yearly growth, India):*

| | 3-year CO2e | 3-year water |
|---|---|---|
| Baseline | 33.4 t | 221 m³ |
| With levers (15% caching, 20% token discipline, 40% right-sizing, 30% moved to Norway) | 12.3 t | 86 m³ |
| Reduction | **63%** | **61%** |

Lever contribution to CO2e: caching 5.0 t, token discipline 4.7 t, right-sizing 6.4 t, cleaner region 5.0 t.

*Uncertainty (chat, medium model, India, 5,000 draws):* carbon P5 = 0.077 kg, median = 0.178 kg, P95 = 0.413 kg per 1,000 requests - a ~5× spread, so single-number footprints should be read as ranges.

## 7. Discussion
- **Model choice and task type dominate.** Right-sizing and avoiding unnecessary image/video generation matter more than polite-prompt trimming.
- **Output length is the bigger prompt lever.** Reading tokens is cheap relative to generating them, so asking for concise answers saves more than deleting filler words.
- **Location matters twice**: it changes carbon *and* water, sometimes in opposite directions.
- **Uncertainty is large**; the tool's value is in *relative* comparison of options under the same assumptions.

## 8. Limitations
Order-of-magnitude assumptions; inference only (no training or embodied carbon); annual-average grid data; synthetic timing curve; heuristic tokeniser; water scarcity not weighted by location. GenAI features: LLM answers can be wrong or incomplete (mitigated by grounding, a judge check and visible context, not eliminated); the footprint of AI calls assumes the provider's model resembles the chosen size class; the live GenAI path was verified with mocked API responses, so run one real call with your own key before demo day.

## 9. Future work
Live grid-intensity API; real tokeniser; measured energy via hardware counters (e.g. CodeCarbon) for open models; water-scarcity weighting; embodied carbon; user-uploaded usage logs; retrieval over a document corpus (e.g. sustainability reports) for the Advisor; evaluating judge reliability against human ratings.

## 10. Conclusion
The project demonstrates that AI's footprint can be made visible, comparable and actionable. By combining carbon, water, uncertainty and intervention testing in one deployable tool, it moves the conversation from "AI is bad/good for the climate" to "which choices reduce impact most, and by how much?"

## 11. References
See `README.md` and the app's Methodology tab (verify each source before citing).

## 12. Viva preparation - likely questions
1. **Why these numbers?** They are editable order-of-magnitude assumptions informed by public research (Luccioni et al. 2023, Li et al. 2023, company disclosures); the tool's value is comparing scenarios consistently.
2. **Why is water included?** Cooling and electricity generation consume water; it can conflict with the carbon-optimal choice.
3. **What is PUE / WUE / EWIF?** Facility-to-IT energy ratio / on-site cooling water per kWh / water used to generate each kWh of electricity.
4. **Why Monte Carlo?** Inputs are uncertain; sampling ranges gives a distribution instead of a falsely precise single value.
5. **Why is output length more important than prompt length?** Generating (decode) a token costs more energy than reading (prefill) one; the model weights input at 15%.
6. **What are the main limitations?** Section 8.
7. **How would you improve it?** Section 9.
8. **Is the Prompt Diet an LLM?** No - deterministic regex rules, so it is free, instant and explainable. The *Smart* Prompt Optimizer in AI Studio is the LLM version, and the two are compared side by side.
9. **How did you test it?** Unit tests for formulas and monotonicity (cleaner grid → lower carbon; levers → lower footprint), mocked-API tests for all three LLM providers, and app tests across all regions and through the AI Studio in demo and live (mocked) mode.
10. **Where is the generative AI?** In AI Studio: a grounded advisor, an LLM prompt optimizer with an LLM-as-judge check, an executive-memo writer, and self-measurement of its own calls.
11. **How do you stop the LLM hallucinating numbers?** The system prompt contains the computed results and assumptions, instructs it to use only those and admit gaps, and the exact context is shown to the user in an expander.
12. **What is prompt injection and how did you handle it?** Untrusted text can try to override instructions. User text is wrapped in tags and declared to be data, the system prompt states that user messages cannot change the rules, and the model has no tools to misuse.
13. **Isn't using an LLM to save energy self-defeating?** Not if the saving exceeds the cost. The optimizer shows the break-even number of uses, so it is applied to reusable prompts.
14. **What happens without an API key?** Demo mode runs deterministic rule-based versions of every feature, clearly labelled.
