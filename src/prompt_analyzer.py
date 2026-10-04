"""Prompt Diet: a rule-based (no API key) prompt efficiency analyser.

It estimates token count, finds filler phrases and repeated sentences,
builds a leaner rewrite and scores the prompt from 0-100.
"""

from __future__ import annotations

import math
import re

# (regex, replacement, human-readable label)
FILLERS: list[tuple[str, str, str]] = [
    (r"\bplease\b,?\s*", "", "please"),
    (r"\bkindly\b\s*", "", "kindly"),
    (r"\bi would like you to\b\s*", "", "I would like you to"),
    (r"\bi want you to\b\s*", "", "I want you to"),
    (r"\bcould you (?:please )?", "", "could you"),
    (r"\bcan you (?:please )?", "", "can you"),
    (r"\bwould you mind\b\s*", "", "would you mind"),
    (r"\bi was wondering if\b\s*", "", "I was wondering if"),
    (r"\bit would be great if\b\s*", "", "it would be great if"),
    (r"\bas an ai(?: language model)?,?\s*", "", "as an AI"),
    (r"\bif possible,?\s*", "", "if possible"),
    (r"\bbasically,?\s*", "", "basically"),
    (r"\bactually,?\s*", "", "actually"),
    (r"\bvery\s+", "", "very"),
    (r"\breally\s+", "", "really"),
    (r"\bjust\s+", "", "just"),
    (r"\bin order to\b", "to", "in order to"),
    (r"\bdue to the fact that\b", "because", "due to the fact that"),
    (r"\bthe fact that\b", "that", "the fact that"),
    (r"\bat this point in time\b", "now", "at this point in time"),
    (r"\bin the event that\b", "if", "in the event that"),
    (r"\bthank you(?: so much)?(?: in advance)?[.!]?", "", "thank you"),
]

_WORD = re.compile(r"\w+(?:'\w+)?")
_PUNCT = re.compile(r"[^\w\s]")


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 1.33 tokens per word + punctuation)."""
    if not text.strip():
        return 0
    words = len(_WORD.findall(text))
    punct = len(_PUNCT.findall(text))
    return max(1, math.ceil(words * 1.33 + punct * 0.9))


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def analyse(text: str) -> dict:
    """Analyse a prompt and return metrics plus a leaner rewrite."""
    tokens = estimate_tokens(text)

    # filler detection
    hits: dict[str, int] = {}
    lean = text
    for pattern, repl, label in FILLERS:
        found = re.findall(pattern, lean, flags=re.IGNORECASE)
        if found:
            hits[label] = hits.get(label, 0) + len(found)
            lean = re.sub(pattern, repl, lean, flags=re.IGNORECASE)

    # repeated sentences
    seen: set[str] = set()
    kept: list[str] = []
    repeats = 0
    for s in _split_sentences(lean):
        key = re.sub(r"\W+", " ", s.lower()).strip()
        if key in seen:
            repeats += 1
            continue
        seen.add(key)
        kept.append(s)
    lean = " ".join(kept)
    lean = re.sub(r"\s{2,}", " ", lean).strip()
    if lean:
        lean = lean[0].upper() + lean[1:]

    lean_tokens = estimate_tokens(lean)
    saved = max(0, tokens - lean_tokens)
    saved_pct = (saved / tokens * 100.0) if tokens else 0.0

    filler_total = sum(hits.values())
    length_penalty = min(15.0, max(0.0, (tokens - 400) / 80.0))
    score = 100.0 - min(60.0, filler_total * 6.0) - min(25.0, repeats * 12.0) - length_penalty
    score = max(0.0, min(100.0, score))

    return {
        "tokens": tokens,
        "lean_text": lean,
        "lean_tokens": lean_tokens,
        "saved_tokens": saved,
        "saved_pct": saved_pct,
        "filler_hits": hits,
        "filler_total": filler_total,
        "repeated_sentences": repeats,
        "score": round(score),
    }


def tips(result: dict) -> list[str]:
    """Plain-language advice based on the analysis."""
    out: list[str] = []
    if result["filler_total"]:
        out.append("Drop polite padding and hedges - models do not need 'please', 'kindly' or 'if possible'.")
    if result["repeated_sentences"]:
        out.append("Remove repeated sentences; say each instruction once.")
    if result["tokens"] > 800:
        out.append("Long prompt: send only the context the task needs (or retrieve relevant chunks).")
    out.append("Ask for the format and length you need (e.g. 'answer in 5 bullet points') - output tokens cost most.")
    out.append("Use the smallest model that does the job; keep the largest for hard tasks.")
    return out
