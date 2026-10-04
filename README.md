# 🌍 GenAI Footprint Lab

**An interactive Streamlit app that estimates the energy, carbon and water cost of generative AI - and tests what actually reduces it.**
<img width="1853" height="917" alt="image" src="https://github.com/user-attachments/assets/69688f61-4b87-4ebd-8e3b-ef2218d095ca" />
<img width="1487" height="852" alt="image" src="https://github.com/user-attachments/assets/3b5ba13d-05a8-4849-8c82-73c19671aac5" />
<img width="1522" height="887" alt="image" src="https://github.com/user-attachments/assets/f34b4367-08e8-437e-a05b-76d9854a36b8" />


Capstone project for the *Artificial Intelligence & Generative AI* module.

> Every AI answer runs on GPUs that draw electricity and evaporate water to stay cool. One request is tiny; billions are not. This lab turns those hidden costs into numbers you can explore, compare and reduce.

No API keys, no external data downloads, no database - it runs fully offline from transparent, editable assumptions.

---

## ✨ Features

| Tab | What it does |
|---|---|
| 🏠 **Overview** | Live headline numbers: footprint of one request by model size, task comparison chart, auto-computed key insights |
| 🤖 **AI Studio** | **Generative AI connected to the app** - grounded Footprint Advisor chat, Smart Prompt Optimizer with LLM-as-judge, Executive Memo writer, and a tracker of the AI calls' own footprint |
| 🧾 **My Footprint Passport** | Enter weekly AI usage → annual energy, carbon, water, everyday equivalents (phone charges, car km, water bottles), personalised "what would reduce it" advice, downloadable report |
| ✂️ **Prompt Diet** | Rule-based prompt analyser (token estimate, filler phrases, repeated sentences, 0-100 efficiency score, leaner rewrite) and the yearly footprint saved |
| 🌐 **Region & Timing** | Compare 9 grids on carbon *and* water (shows the carbon-vs-water trade-off), plus carbon-aware scheduling simulation |
| 🏢 **Organisation Planner** | 3-year company projection with four levers (caching, token discipline, model right-sizing, cleaner region) and a waterfall showing what each saves |
| 🎲 **Uncertainty Lab** | Monte Carlo simulation (P5 / median / P95) and tornado sensitivity charts - shows how confident the estimates are |
| 📚 **Methodology** | Formulas, every assumption table, limitations, ethics note, references |

### 🤖 The generative-AI part

| Feature | How the LLM is used |
|---|---|
| **Footprint Advisor** | Chat grounded in the app's assumptions and *your* computed results (context injection), told not to invent numbers |
| **Smart Prompt Optimizer** | LLM shortens a prompt; a second LLM call (**LLM-as-judge**) scores whether meaning was preserved (1-5); the app computes CO2/water saved and the **break-even number of uses** |
| **Executive Memo** | Planner results in, a plain-language memo for decision-makers out |
| **Self-measurement** | Token usage reported by the provider is converted to the AI calls' own energy, carbon and water |

Works with **Anthropic Claude, OpenAI or Google Gemini** (plain HTTPS, no SDKs). With no key it runs in **Demo mode** using rule-based versions of every feature, so it always works for reviewers.

**Using a real key**
- *Locally / on the deployed app:* choose a provider in the sidebar and paste your key (kept only for your session).
- *Preconfigured (recommended for deployment):* copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` or `GEMINI_API_KEY`. On Streamlit Cloud paste the same lines into **App settings → Secrets**. Never commit real keys (`secrets.toml` is git-ignored).
- Model names are editable in the sidebar; defaults are low-cost models. If a default is retired, enter a current model name from your provider.
- Safeguards: input length limit, 40 AI calls per session, friendly errors, keys never stored or logged.

### What makes it different
- Covers **water** as well as carbon, and exposes the trade-off between them.
- Treats numbers as **ranges** (Monte Carlo + sensitivity), not false precision.
- Tests **interventions**, not just measurements.
- Includes an **India grid** view alongside global regions.

---

## 🗂️ Project structure

```
genai_footprint_lab/
├── app.py                  # Streamlit app (UI only)
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
├── .streamlit/
│   ├── config.toml         # theme + headless server settings
│   └── secrets.toml.example  # template for API keys (copy to secrets.toml)
├── src/
│   ├── __init__.py
│   ├── data.py             # ALL editable assumptions (models, regions, tasks, references)
│   ├── llm.py              # provider-agnostic LLM client (Claude / OpenAI / Gemini)
│   ├── prompts.py          # system prompts, grounding-context builder, judge parser
│   ├── fallback.py         # rule-based Demo mode for every AI feature
│   ├── calculator.py       # tokens -> energy -> carbon & water; equivalents; passport
│   ├── prompt_analyzer.py  # Prompt Diet engine (rule-based)
│   ├── scenarios.py        # org planner, Monte Carlo, sensitivity, carbon-aware timing
│   ├── charts.py           # Plotly chart builders
│   ├── report.py           # markdown report export
│   └── utils.py            # number/unit formatting
├── tests/
│   ├── test_core.py        # formula, scenario and prompt-analyser unit tests
│   ├── test_llm.py         # mocked-API tests for all three providers + fallbacks
│   └── test_app_smoke.py   # renders the app in every region; drives AI Studio (demo + mocked live)
└── docs/
    ├── STEP_BY_STEP_GUIDE.md  # how to run, check results, deploy, impact
    ├── PROJECT_REPORT.md   # full report: problem, method, results, viva Q&A
    └── PRESENTATION_OUTLINE.md
```

---

## 🚀 Quick start

```bash
# 1. clone
git clone https://github.com/<your-username>/genai-footprint-lab.git
cd genai-footprint-lab

# 2. (recommended) virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. install
pip install -r requirements.txt

# 4. run
streamlit run app.py
```

The app opens at http://localhost:8501.
                 [https://gen-aifootprintlab.streamlit.app/](https://gen-aifootprintlab.streamlit.app/)

### Run the tests
```bash
pip install pytest
pytest -q
```

---

## ☁️ Deploy on Streamlit Community Cloud (free)

1. Push this folder to a **public GitHub repository** (see below).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **Create app → Deploy a public app from GitHub**.
4. Choose your repository, branch `main`, and main file path `app.py`.
5. Click **Deploy**. Streamlit installs `requirements.txt` automatically and gives you a public URL.
6. Paste that URL into your README and your submission.

### Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: GenAI Footprint Lab"
git branch -M main
git remote add origin https://github.com/<your-username>/genai-footprint-lab.git
git push -u origin main
```

---

## 🔧 Customising the assumptions

Everything numeric lives in `src/data.py`:

- `MODEL_CLASSES` - Wh per 1,000 tokens / per image / per video clip
- `REGIONS` - grid carbon intensity, PUE, on-site water (WUE), electricity water (EWIF)
- `TASKS` - input/output tokens per task
- `INPUT_TOKEN_WEIGHT`, `REASONING_TOKEN_MULTIPLIER`

Change a value and the whole app recalculates. To add a region, add one dictionary entry.

---

## 🧮 Method in one minute

```
E_IT   = (w_model / 1000) × (T_out × k_reason + α × T_in)
E_fac  = E_IT × PUE
CO2e   = E_fac × grid carbon intensity
Water  = E_IT × WUE  +  E_fac × EWIF
```

`w_model` = Wh per 1,000 generated tokens, `α` = relative cost of reading a token, `k_reason` = hidden-token multiplier in reasoning mode. Full details and limitations are in the app's **Methodology** tab and `docs/PROJECT_REPORT.md`.

---

## ⚠️ Limitations

- Values are **order-of-magnitude assumptions** informed by public research, not measurements of any specific product. Verify against current sources before quoting them.
- Inference only - training and hardware manufacturing (embodied carbon) are excluded.
- Annual-average grid intensity; the timing chart uses a **synthetic** 24-hour curve, not live grid data.
- Token counts use a heuristic, not a real tokeniser.
- LLM answers can be wrong; the Advisor is grounded and shows its context, but check numbers against the dashboards.
- The live-AI path is tested with mocked API responses; try one real call with your key before presenting.

## 📚 References that informed the assumptions
- Luccioni, Jernite & Strubell (2023). *Power Hungry Processing.* arXiv:2311.16863
- Li, Yang, Islam & Ren (2023). *Making AI Less "Thirsty".* arXiv:2304.03271
- Patterson et al. (2021). *Carbon Emissions and Large Neural Network Training.* arXiv:2104.10350
- Strubell, Ganesh & McCallum (2019). *Energy and Policy Considerations for Deep Learning in NLP.*
- Google (2025). *Measuring the environmental impact of delivering AI at Google scale.*
- IEA (2025). *Energy and AI.*
- Central Electricity Authority, India - *CO2 Baseline Database for the Indian Power Sector.*

## 📄 License
MIT - see `LICENSE`.

**AUTHOR**
**Sadia Uzma Ashrafi**
