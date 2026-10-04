"""Smoke test: the whole app renders without exceptions. Run with: pytest -q"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_renders_default():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, [e.value for e in at.exception]


def test_app_renders_every_region():
    at = AppTest.from_file(APP, default_timeout=60).run()
    from src.data import REGIONS
    for region in REGIONS:
        at.sidebar.selectbox[0].set_value(region).run()
        assert not at.exception, (region, [e.value for e in at.exception])


def test_ai_studio_demo_mode_flows():
    at = AppTest.from_file(APP, default_timeout=60).run()
    # chat in demo mode
    at.chat_input[0].set_value("Which region has the lowest carbon?").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Demo mode" in m.value for m in at.markdown)
    # optimiser + memo buttons in demo mode
    next(b for b in at.button if b.key == "ai_opt_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    next(b for b in at.button if b.key == "ai_memo_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]


def test_ai_studio_live_mode_with_mocked_api(monkeypatch):
    from src import llm

    class R:
        status_code = 200

        def __init__(self, text):
            self._t = text

        def json(self):
            return {"content": [{"type": "text", "text": self._t}],
                    "usage": {"input_tokens": 400, "output_tokens": 120}}

    def fake_post(url, headers=None, json=None, timeout=None):
        sys_prompt = json.get("system", "")
        if "ORIGINAL" in json["messages"][0]["content"]:
            return R("SCORE: 5\nREASON: Everything preserved.")
        if "rewrite prompts" in sys_prompt:
            return R("Write a simple bullet-point exam summary.")
        return R("Use smaller models first.")

    monkeypatch.setattr(llm.requests, "post", fake_post)
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.sidebar.selectbox[1].set_value("Anthropic Claude").run()
    at.sidebar.text_input[0].set_value("test-key").run()
    at.chat_input[0].set_value("How do I cut my footprint?").run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Use smaller models first." in m.value for m in at.markdown)
    next(b for b in at.button if b.key == "ai_opt_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("5/5" in str(m.value) for m in at.metric)
    next(b for b in at.button if b.key == "ai_memo_btn").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.session_state["ai_calls"]) == 4  # advisor + optimizer + judge + memo
