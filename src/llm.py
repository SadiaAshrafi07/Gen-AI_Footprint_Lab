"""Thin, provider-agnostic LLM client (no SDK needed - plain HTTPS via requests).

Supports Anthropic Claude, OpenAI and Google Gemini. Keys are never logged or
stored; they are passed per call and sent only to the provider's own endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from .prompt_analyzer import estimate_tokens

DEMO = "Demo mode (no API key)"

PROVIDERS: dict[str, dict] = {
    "Anthropic Claude": {
        "id": "anthropic",
        "default_model": "claude-haiku-4-5-20251001",
        "secret": "ANTHROPIC_API_KEY",
        "url": "https://api.anthropic.com/v1/messages",
    },
    "OpenAI": {
        "id": "openai",
        "default_model": "gpt-4o-mini",
        "secret": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
    },
    "Google Gemini": {
        "id": "gemini",
        "default_model": "gemini-2.0-flash",
        "secret": "GEMINI_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
    },
}

MAX_INPUT_CHARS = 6000


class LLMError(Exception):
    """Raised with a user-friendly message when an LLM call fails."""


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    model: str
    max_tokens: int = 700
    temperature: float = 0.3


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


def _friendly_error(status: int, body: str) -> str:
    if status in (401, 403):
        return "The provider rejected the API key (check the key and that billing/quota is enabled)."
    if status == 404:
        return "Model not found - check the model name in the sidebar."
    if status == 429:
        return "Rate limit or quota reached - wait a moment or check your plan."
    if status >= 500:
        return "The provider is temporarily unavailable - try again shortly."
    return f"The provider returned an error (HTTP {status}). {body[:160]}"


def _extract_error_message(resp: requests.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error", data)
        if isinstance(err, dict):
            return str(err.get("message", ""))[:160]
        return str(err)[:160]
    except Exception:  # noqa: BLE001
        return ""


def call_llm(
    cfg: LLMConfig,
    system: str,
    messages: list[dict],
    timeout: int = 45,
) -> LLMResult:
    """Send a chat request and return text plus token usage.

    `messages` is a list of {"role": "user"|"assistant", "content": str}.
    """
    if cfg.provider not in PROVIDERS:
        raise LLMError("Unknown provider.")
    if not cfg.api_key.strip():
        raise LLMError("No API key provided.")
    for m in messages:
        if len(m["content"]) > MAX_INPUT_CHARS:
            raise LLMError(f"Input too long (limit {MAX_INPUT_CHARS} characters).")

    p = PROVIDERS[cfg.provider]
    kind = p["id"]
    try:
        if kind == "anthropic":
            resp = requests.post(
                p["url"],
                headers={
                    "x-api-key": cfg.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": cfg.model,
                    "max_tokens": cfg.max_tokens,
                    "temperature": cfg.temperature,
                    "system": system,
                    "messages": messages,
                },
                timeout=timeout,
            )
        elif kind == "openai":
            resp = requests.post(
                p["url"],
                headers={"Authorization": f"Bearer {cfg.api_key}", "content-type": "application/json"},
                json={
                    "model": cfg.model,
                    "max_tokens": cfg.max_tokens,
                    "temperature": cfg.temperature,
                    "messages": [{"role": "system", "content": system}, *messages],
                },
                timeout=timeout,
            )
        else:  # gemini
            contents = [
                {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                for m in messages
            ]
            resp = requests.post(
                p["url"].format(model=cfg.model),
                headers={"x-goog-api-key": cfg.api_key, "content-type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": contents,
                    "generationConfig": {
                        "maxOutputTokens": cfg.max_tokens,
                        "temperature": cfg.temperature,
                    },
                },
                timeout=timeout,
            )
    except requests.RequestException as exc:
        raise LLMError(f"Network problem reaching the provider ({type(exc).__name__}).") from exc

    if resp.status_code != 200:
        raise LLMError(_friendly_error(resp.status_code, _extract_error_message(resp)))

    try:
        data = resp.json()
        if kind == "anthropic":
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            usage = data.get("usage", {})
            tin, tout = usage.get("input_tokens"), usage.get("output_tokens")
        elif kind == "openai":
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage", {})
            tin, tout = usage.get("prompt_tokens"), usage.get("completion_tokens")
        else:
            text = "".join(part.get("text", "") for part in data["candidates"][0]["content"]["parts"])
            usage = data.get("usageMetadata", {})
            tin, tout = usage.get("promptTokenCount"), usage.get("candidatesTokenCount")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise LLMError("Unexpected response format from the provider.") from exc

    if not text.strip():
        raise LLMError("The model returned an empty answer.")

    if tin is None:
        tin = estimate_tokens(system) + sum(estimate_tokens(m["content"]) for m in messages)
    if tout is None:
        tout = estimate_tokens(text)
    return LLMResult(text=text.strip(), input_tokens=int(tin), output_tokens=int(tout))
