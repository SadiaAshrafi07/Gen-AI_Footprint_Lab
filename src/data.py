"""Reference data and editable assumptions for the GenAI Footprint Lab.

IMPORTANT: every number here is an *order-of-magnitude assumption* informed by
public research and company disclosures. Real values vary by hardware, batching,
data-centre design and year. Edit the dictionaries below to test your own
assumptions - the whole app recalculates from them.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Model classes: IT-equipment energy per unit of work (before facility overhead)
# --------------------------------------------------------------------------- #
MODEL_CLASSES: dict[str, dict] = {
    "small": {
        "label": "Small (~8B params)",
        "text_wh_per_1k_tokens": 0.06,
        "image_wh": 0.9,
        "video_wh": 150.0,  # per 5-second clip, low confidence
    },
    "medium": {
        "label": "Medium (~70B params)",
        "text_wh_per_1k_tokens": 0.25,
        "image_wh": 2.9,
        "video_wh": 400.0,
    },
    "large": {
        "label": "Large / frontier",
        "text_wh_per_1k_tokens": 0.60,
        "image_wh": 5.0,
        "video_wh": 940.0,
    },
}

# Reading (prefill) a token costs less than generating one (decode).
INPUT_TOKEN_WEIGHT = 0.15

# "Reasoning" / chain-of-thought modes generate many hidden tokens.
REASONING_TOKEN_MULTIPLIER = 8.0

# --------------------------------------------------------------------------- #
# Task catalogue
# --------------------------------------------------------------------------- #
TASKS: dict[str, dict] = {
    "quick_qa": {"label": "Quick Q&A", "kind": "text", "in": 100, "out": 250},
    "detailed_chat": {"label": "Detailed chat answer", "kind": "text", "in": 300, "out": 600},
    "code_gen": {"label": "Code generation", "kind": "text", "in": 400, "out": 900},
    "summarise_doc": {"label": "Summarise a long document", "kind": "text", "in": 4000, "out": 400},
    "draft_report": {"label": "Draft an essay / report", "kind": "text", "in": 500, "out": 2000},
    "translation": {"label": "Translate a page", "kind": "text", "in": 600, "out": 700},
    "rag_answer": {"label": "Document Q&A (RAG, 5 chunks)", "kind": "text", "in": 3000, "out": 500},
    "image": {"label": "Generate 1 image", "kind": "image"},
    "video": {"label": "Generate 5-second video clip", "kind": "video"},
}

# --------------------------------------------------------------------------- #
# Regions: grid carbon intensity and water factors
#   ci   = grid carbon intensity, gCO2e per kWh
#   pue  = data-centre Power Usage Effectiveness (facility energy / IT energy)
#   wue  = on-site cooling water, litres per kWh of IT energy
#   ewif = off-site water used to generate electricity, litres per kWh
# --------------------------------------------------------------------------- #
REGIONS: dict[str, dict] = {
    "India (national grid)": {"ci": 710, "pue": 1.50, "wue": 1.80, "ewif": 3.5},
    "USA (average)": {"ci": 370, "pue": 1.30, "wue": 0.55, "ewif": 3.1},
    "European Union (average)": {"ci": 250, "pue": 1.30, "wue": 0.40, "ewif": 1.8},
    "France (nuclear-heavy)": {"ci": 55, "pue": 1.30, "wue": 0.40, "ewif": 2.5},
    "Norway (hydro)": {"ci": 30, "pue": 1.15, "wue": 0.10, "ewif": 1.0},
    "China (national grid)": {"ci": 580, "pue": 1.50, "wue": 1.20, "ewif": 3.0},
    "Australia": {"ci": 550, "pue": 1.40, "wue": 0.80, "ewif": 2.8},
    "Singapore": {"ci": 470, "pue": 1.40, "wue": 1.00, "ewif": 1.4},
    "Brazil (hydro-heavy)": {"ci": 100, "pue": 1.40, "wue": 1.00, "ewif": 5.0},
}

DEFAULT_REGION = "India (national grid)"

# --------------------------------------------------------------------------- #
# Everyday-equivalent constants
# --------------------------------------------------------------------------- #
PHONE_CHARGE_WH = 12.0
LED_BULB_WATTS = 10.0
CAR_G_CO2_PER_KM = 120.0
TREE_KG_CO2_PER_YEAR = 21.0
BOTTLE_ML = 500.0
SHOWER_LITRES = 60.0

# --------------------------------------------------------------------------- #
# Literature the assumptions are informed by (verify before citing in a report)
# --------------------------------------------------------------------------- #
REFERENCES = [
    "Luccioni, Jernite & Strubell (2023). Power Hungry Processing: Watts Driving the Cost of AI Deployment? arXiv:2311.16863",
    "Li, Yang, Islam & Ren (2023). Making AI Less 'Thirsty'. arXiv:2304.03271",
    "Patterson et al. (2021). Carbon Emissions and Large Neural Network Training. arXiv:2104.10350",
    "Strubell, Ganesh & McCallum (2019). Energy and Policy Considerations for Deep Learning in NLP. ACL",
    "Google (2025). Measuring the environmental impact of delivering AI at Google scale",
    "International Energy Agency (2025). Energy and AI",
    "Central Electricity Authority, India. CO2 Baseline Database for the Indian Power Sector",
]
