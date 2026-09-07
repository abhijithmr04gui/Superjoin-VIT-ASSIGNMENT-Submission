"""
Thin wrapper around the Anthropic Messages API.

Every call site asks for structured JSON and this module is responsible
for: making the call, stripping markdown fences, parsing JSON, and
retrying once with a "your last response was invalid JSON" correction
message. It does NOT decide what to do if parsing still fails after
retry - callers must handle that (usually: record an ExtractionFailure
and move on, per assignment section 18).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import anthropic

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("llm_client")

_client: anthropic.Anthropic | None = None


class LLMError(Exception):
    pass


class LLMNotConfiguredError(LLMError):
    pass


def _get_client() -> anthropic.Anthropic:
    global _client
    if not settings.anthropic_api_key:
        raise LLMNotConfiguredError("ANTHROPIC_API_KEY is not set - see .env.example")
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


@dataclass
class LLMJsonResult:
    ok: bool
    data: dict | list | None
    raw_text: str
    error: str | None = None


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_block(text: str) -> str:
    """Grab the first {...} or [...] block if the model added prose around it."""
    stripped = _strip_fences(text)
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\{.*\}|\[.*\])", stripped, re.DOTALL)
    if match:
        return match.group(1)
    return stripped


def call_json(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 2000,
    temperature: float = 0.0,
) -> LLMJsonResult:
    """
    Call the LLM expecting a single JSON object/array back. Retries once
    with a correction message if the first response fails to parse.
    """
    client = _get_client()
    messages = [{"role": "user", "content": user_prompt}]

    for attempt in range(2):
        try:
            response = client.messages.create(
                model=settings.anthropic_model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=messages,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM API call failed: %s", exc)
            return LLMJsonResult(ok=False, data=None, raw_text="", error=str(exc))

        raw_text = "".join(block.text for block in response.content if block.type == "text")

        try:
            candidate = _extract_json_block(raw_text)
            data = json.loads(candidate)
            return LLMJsonResult(ok=True, data=data, raw_text=raw_text)
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned invalid JSON (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                messages.append({"role": "assistant", "content": raw_text})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid JSON "
                            f"(error: {exc}). Reply again with ONLY valid JSON, "
                            "no markdown fences, no commentary."
                        ),
                    }
                )
                continue
            return LLMJsonResult(ok=False, data=None, raw_text=raw_text, error=f"invalid JSON: {exc}")

    return LLMJsonResult(ok=False, data=None, raw_text="", error="exhausted retries")
