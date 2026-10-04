"""Tests for the GenAI integration. All network calls are mocked - no API key needed."""

import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import llm  # noqa: E402
from src.calculator import call_footprint  # noqa: E402
from src.fallback import demo_advisor, demo_summary  # noqa: E402
from src.llm import LLMConfig, LLMError, call_llm  # noqa: E402
from src.prompts import build_context, parse_judge  # noqa: E402

MSGS = [{"role": "user", "content": "hello"}]


class FakeResp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _patch(monkeypatch, resp, capture=None):
    def fake_post(url, headers=None, json=None, timeout=None):
        if capture is not None:
            capture.update(url=url, headers=headers, json=json)
        return resp

    monkeypatch.setattr(llm.requests, "post", fake_post)


def test_anthropic_success_and_request_shape(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeResp(200, {"content": [{"type": "text", "text": "Hi!"}],
                                       "usage": {"input_tokens": 11, "output_tokens": 3}}), cap)
    out = call_llm(LLMConfig("Anthropic Claude", "k", "m"), "sys", MSGS)
    assert (out.text, out.input_tokens, out.output_tokens) == ("Hi!", 11, 3)
    assert cap["headers"]["x-api-key"] == "k"
    assert cap["json"]["system"] == "sys" and cap["json"]["messages"] == MSGS


def test_openai_success(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeResp(200, {"choices": [{"message": {"content": "Yo"}}],
                                       "usage": {"prompt_tokens": 7, "completion_tokens": 2}}), cap)
    out = call_llm(LLMConfig("OpenAI", "k", "m"), "sys", MSGS)
    assert (out.text, out.input_tokens, out.output_tokens) == ("Yo", 7, 2)
    assert cap["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert cap["headers"]["Authorization"] == "Bearer k"


def test_gemini_success_key_not_in_url(monkeypatch):
    cap = {}
    _patch(monkeypatch, FakeResp(200, {"candidates": [{"content": {"parts": [{"text": "Hey"}]}}],
                                       "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 1}}), cap)
    out = call_llm(LLMConfig("Google Gemini", "SECRETKEY", "gemini-x"), "sys", MSGS + [{"role": "assistant", "content": "a"}])
    assert out.text == "Hey" and out.input_tokens == 5
    assert "SECRETKEY" not in cap["url"]  # key travels in a header, not the URL
    assert cap["json"]["contents"][-1]["role"] == "model"


@pytest.mark.parametrize("status,needle", [(401, "key"), (404, "Model not found"), (429, "Rate limit"), (503, "unavailable")])
def test_http_errors_are_friendly(monkeypatch, status, needle):
    _patch(monkeypatch, FakeResp(status, {"error": {"message": "x"}}))
    with pytest.raises(LLMError) as e:
        call_llm(LLMConfig("OpenAI", "k", "m"), "s", MSGS)
    assert needle in str(e.value)
    assert "k" != str(e.value)


def test_network_error_does_not_leak_key(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("https://x?key=SECRETKEY")

    monkeypatch.setattr(llm.requests, "post", boom)
    with pytest.raises(LLMError) as e:
        call_llm(LLMConfig("OpenAI", "SECRETKEY", "m"), "s", MSGS)
    assert "SECRETKEY" not in str(e.value)


def test_validation_errors():
    with pytest.raises(LLMError):
        call_llm(LLMConfig("OpenAI", "  ", "m"), "s", MSGS)
    with pytest.raises(LLMError):
        call_llm(LLMConfig("Nope", "k", "m"), "s", MSGS)
    with pytest.raises(LLMError):
        call_llm(LLMConfig("OpenAI", "k", "m"), "s", [{"role": "user", "content": "x" * 7000}])


def test_bad_and_empty_responses(monkeypatch):
    _patch(monkeypatch, FakeResp(200, {"unexpected": True}))
    with pytest.raises(LLMError):
        call_llm(LLMConfig("OpenAI", "k", "m"), "s", MSGS)
    _patch(monkeypatch, FakeResp(200, {"choices": [{"message": {"content": "  "}}]}))
    with pytest.raises(LLMError):
        call_llm(LLMConfig("OpenAI", "k", "m"), "s", MSGS)


def test_missing_usage_is_estimated(monkeypatch):
    _patch(monkeypatch, FakeResp(200, {"choices": [{"message": {"content": "some answer text"}}]}))
    out = call_llm(LLMConfig("OpenAI", "k", "m"), "system prompt", MSGS)
    assert out.input_tokens > 0 and out.output_tokens > 0


def test_call_footprint_scales_with_tokens():
    small = call_footprint(100, 100, "medium", "India (national grid)")
    big = call_footprint(1000, 1000, "medium", "India (national grid)")
    assert big.co2_g > small.co2_g > 0


def test_parse_judge():
    assert parse_judge("SCORE: 4\nREASON: kept everything") == (4, "kept everything")
    score, reason = parse_judge("garbled")
    assert score is None and reason


def test_context_contains_grounding_data():
    passport = {"energy_wh": 5000, "co2_g": 2000, "water_ml": 9000, "model_label": "Medium (~70B params)",
                "reasoning_share": 0.1,
                "breakdown": [{"Activity": "Quick Q&A", "Per week": 20, "CO2e (kg/yr)": 1.2}],
                "insights": ["Use small models."]}
    ctx = build_context("India (national grid)", passport, None)
    assert "710" in ctx and "Quick Q&A" in ctx and "Use small models." in ctx


def test_demo_fallbacks_return_text():
    ans = demo_advisor("Which region is best and how much water?", "India (national grid)", None, None)
    assert "Demo mode" in ans and len(ans) > 50
    org = {"employees": 100, "adoption": 0.5, "years": 3, "region": "India (national grid)", "base_co2": 10.0,
           "opt_co2": 5.0, "base_water": 100.0, "opt_water": 60.0, "co2_cut": 50.0, "water_cut": 40.0,
           "cache": 0.1, "trim": 0.1, "rightsize": 0.2, "shift": 0.3, "target": "Norway (hydro)",
           "waterfall": [{"step": "Baseline", "co2_t": 10.0}, {"step": "Caching", "co2_t": 9.0},
                         {"step": "Right-sizing", "co2_t": 7.0}, {"step": "Cleaner region", "co2_t": 5.0}]}
    memo = demo_summary(org)
    assert "50%" in memo and "Right-sizing" in memo
