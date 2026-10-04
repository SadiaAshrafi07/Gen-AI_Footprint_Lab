# GenAI Footprint Lab - Step-by-Step Guide

What it is: a Streamlit web app that estimates the **energy, carbon and water** cost of generative AI, tests ways to reduce it, and has a **built-in LLM (AI Studio)** that advises, optimises prompts and writes memos.

---

## Part 1 - Where to run it

| Option | Use it for | Needs |
|---|---|---|
| **Your own laptop** (Windows / Mac / Linux) | Building, testing, taking screenshots | Python 3.10+ |
| **Streamlit Community Cloud** (free, online) | Final public link for submission | GitHub account |
| ~~Claude chat window~~ | Not possible - the app must run on your machine or Streamlit Cloud | - |

Do Part 2 first (laptop), then Part 7 (deploy).

---

## Part 2 - Run it on your laptop

1. **Install Python 3.10 or newer** from python.org. On Windows tick **"Add Python to PATH"**. Check: open a terminal and run `python --version`.
2. **Unzip** `genai_footprint_lab.zip`. You get a folder `genai_footprint_lab`.
3. **Open a terminal inside that folder.** Windows: open the folder in File Explorer, type `cmd` in the address bar, press Enter. Mac/Linux: `cd` into the folder. VS Code: File → Open Folder, then Terminal → New Terminal.
4. **Create a virtual environment** (keeps packages tidy):
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
5. **Install packages:** `pip install -r requirements.txt`
6. **Start the app:** `streamlit run app.py`
   If `streamlit` is "not recognized", use `python -m streamlit run app.py`.
7. Your browser opens at **http://localhost:8501**. Leave the terminal open. Stop the app with `Ctrl + C`.

Run from the project folder, otherwise Python cannot find the `src` package.

---

## Part 3 - Check that it works (2 minutes)

In a second terminal (same folder, venv active):

```
pip install pytest
pytest -q
```

Expected: **`32 passed`**. These tests check the formulas, all three LLM providers (with fake API replies), and that every tab loads in every region.

---

## Part 4 - Walk through every tab and check the results

Defaults: region **India (national grid)**. The numbers below are what you should see on a fresh start, so you can confirm it is working.

| # | Tab | What to do | What you should see |
|---|---|---|---|
| 1 | **Overview** | Just read it | Medium model, one chat answer: **0.242 Wh, 0.172 g CO2e, 1.14 mL water**. Insights: image ≈ **18x** a chat answer, video ≈ **2,500x**, India vs Norway ≈ **31x** more CO2e |
| 2 | **My Footprint Passport** | Leave defaults, then change "Generate 1 image" from 3 to 0 | Defaults: **1.54 kWh, 1.09 kg CO2e, 7.21 L** per year. Images are the biggest source (0.48 kg). After setting images to 0 the total drops sharply. "What would reduce it" lists small models (~73% less carbon) and a cleaner region |
| 3 | **Prompt Diet** | Use the sample prompt | Score **28/100**, **84 → 40 tokens**, **13 filler phrases**, 1 repeated sentence. The rewrite is rough on purpose - it is rule-based; the AI version in AI Studio is cleaner |
| 4 | **Region & Timing** | Switch the sidebar region to *Norway* and back | Carbon bars change massively; the scatter shows **carbon vs water** (the cleanest grid is not always the driest). Timing: shifting 40% of work to the 4 cleanest hours saves **≈10%** |
| 5 | **Organisation Planner** | Leave defaults | **33.4 t → 12.3 t CO2e (-63%)** and **221 → 86 m³ water (-61%)** over 3 years. Waterfall shows what each lever saves. Move a lever slider to 0 and watch that step vanish |
| 6 | **Uncertainty Lab** | Leave defaults | Carbon per 1,000 chat answers: **P5 ≈ 0.077 kg, median ≈ 0.178 kg, P95 ≈ 0.413 kg** (about 5x spread). Tornado charts show which assumption matters most |
| 7 | **Methodology** | Read | Formulas, every assumption, limitations |

### Verify one number by hand (great for your viva)
One "Detailed chat answer", medium model, India:

- IT energy = 0.25 Wh/1k × (600 output + 0.15 × 300 input) / 1000 = **0.161 Wh**
- Facility energy = 0.161 × PUE 1.5 = **0.242 Wh**
- Carbon = 0.242 / 1000 × 710 g/kWh = **0.172 g**
- Water = 0.161/1000 × 1.8 + 0.242/1000 × 3.5 = **1.14 mL**

It matches the Overview tab, so the app is calculating what the formulas say.

---

## Part 5 - Use the Generative-AI part (AI Studio)

### 5a. Demo mode (no key, works immediately)
1. Open **🤖 AI Studio**. The blue box says *Demo mode*.
2. **Footprint Advisor:** click a suggested question or type one. You get a rule-based answer marked "Demo mode".
3. **Smart Prompt Optimizer:** click **Optimise prompt** → rule-based rewrite and footprint saved.
4. **Executive Memo:** click **Generate executive memo** → a memo built from your Organisation Planner numbers.

### 5b. Live generative AI (with an API key)
1. Get a key from a provider you have access to: Anthropic Claude, OpenAI, or Google Gemini (Gemini and some others offer free tiers; check each provider's current terms).
2. In the **sidebar → GenAI provider**, choose the provider, paste the key. Keep the default model name or type a current one if you get "Model not found".
3. Success box turns green: *Live GenAI: provider · model*.
4. Try, in order:
   - **Advisor:** "What is the single biggest way for me to cut my footprint?" → should cite your numbers (e.g. images, model size). Open *What the model is given* to see the grounding context.
   - **Optimizer:** click Optimise prompt → clean AI rewrite, an **LLM-judge score (1-5)**, CO2/water saved per 1,000 uses, and the **break-even number of uses**.
   - **Executive Memo:** a ~150-word memo using only your planner numbers.
   - **📡 The AI's own footprint:** shows calls made, tokens used, and the energy/carbon/water of the AI calls themselves.
5. Never put a real key in GitHub. For deployment use Streamlit **Secrets** (Part 7).

How to judge whether the GenAI answers are good: numbers in the answer should match the dashboards; a judge score of 4-5 means the rewrite kept all requirements; if it scores ≤3 the app warns you.

---

## Part 6 - Customise before submitting (makes it yours)

- Edit `src/data.py` to change assumptions (a region, model energy, task token counts). Everything recalculates.
- Change the sample prompt or add your own region (e.g. a state-level India grid) and note it in your report.
- Put your name and college in the README header.
- Run the app again plus `pytest -q` after any change.

---

## Part 7 - Put it online (free)

1. Create a GitHub account → **New repository** → name `genai-footprint-lab`, **Public**.
2. In the project folder terminal:
   ```
   git init
   git add .
   git commit -m "GenAI Footprint Lab capstone"
   git branch -M main
   git remote add origin https://github.com/<your-username>/genai-footprint-lab.git
   git push -u origin main
   ```
3. Go to **share.streamlit.io**, sign in with GitHub → **Create app** → pick the repo, branch `main`, main file `app.py` → **Deploy**.
4. (Optional live AI) App → **Settings → Secrets**, paste:
   ```
   ANTHROPIC_API_KEY = "your-key"
   ```
   (or `OPENAI_API_KEY` / `GEMINI_API_KEY`). Your key is then used for everyone visiting, so use a key with a spending limit, or leave it out and let the app run in Demo mode.
5. Copy the public URL (looks like `https://<name>.streamlit.app`) into your submission.

---

## Part 8 - What impact does it make?

**What it does for people**
- **Individuals** see their yearly AI footprint in everyday terms and learn which habit matters most (in the default case: image generation, then chat; switching to small models cuts ~73% of carbon).
- **Organisations** can plan and compare reduction levers before acting. Default example (fictional 5,000-employee company): **−63% carbon (≈21 t CO2e) and −61% water (≈135 m³) over 3 years**. In everyday terms that is roughly the emissions of 175,000 km of petrol-car driving and about 2,200 showers of water.
- **Students / decision-makers** learn that location, model size and output length matter, and that carbon and water can pull in opposite directions.
- **Prompt writers** get a measurable, break-even-checked way to shorten reusable prompts.

**Be honest about the limits (examiners respect this)**
- The impact figures are **modelled estimates** from editable, order-of-magnitude assumptions, not measurements of a real company.
- The app does not cut emissions by itself; its impact is **awareness and better decisions**.
- LLM answers can be wrong; that is why the Advisor is grounded, shows its context, and the judge checks rewrites.

**Why it should score well**
| Likely criterion | What you can point to |
|---|---|
| Originality | Unusual topic: AI's own water + carbon footprint, with India grid |
| Technical depth | Energy model, Monte Carlo, sensitivity, scenario planner |
| GenAI integration | Grounded advisor, LLM prompt optimizer + LLM-as-judge, memo writer, self-measurement |
| Responsible AI | Uncertainty shown, limitations stated, prompt-injection and key-safety handling |
| Engineering | Modular code, 32 tests, demo fallback, deployed app |
| Documentation | README, PROJECT_REPORT, this guide |

---

## Part 9 - Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` not recognized | `python -m streamlit run app.py` |
| `ModuleNotFoundError: src` | You are not in the project folder; `cd` into it |
| `ModuleNotFoundError: plotly` etc. | Activate the venv, then `pip install -r requirements.txt` |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
| "API key rejected" | Re-copy the key; check billing/quota on the provider site |
| "Model not found" | Type a current model name in the sidebar |
| "Rate limit reached" | Wait a minute or use Demo mode |
| Deployed app shows an import error | Make sure `requirements.txt` and the `src/` folder are in the repo root |

---

## Part 10 - Submission checklist

- [ ] App runs locally and `pytest -q` shows 32 passed
- [ ] You tried **one real AI call** with your own key (all features)
- [ ] Repo is public with README; no API keys committed
- [ ] App deployed; public URL works in a private browser window
- [ ] Screenshots taken from your own run
- [ ] PPT written **by you, in your own words** (outline in `docs/PRESENTATION_OUTLINE.md`)
- [ ] You can explain: the formulas, grounding, LLM-as-judge, prompt injection, limitations (viva list in `docs/PROJECT_REPORT.md`)
- [ ] Capstone uploaded, then take the Module End Test
