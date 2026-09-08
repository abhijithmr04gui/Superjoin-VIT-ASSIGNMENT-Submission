"""
Thin wrapper around the Gemini API (google-genai).

Every call site asks for structured JSON and this module is responsible
for: making the call, stripping markdown fences, parsing JSON, and
retrying once with a "your last response was invalid JSON" correction
message. It does NOT decide what to do if parsing still fails after
retry - callers must handle that (usually: record an ExtractionFailure
and move on, per assignment section 18).

Rate limit management lives here too, centrally, since every LLM call
site (extraction, comparison, entity resolution) goes through
`call_json`:

  - A shared, thread-safe rate limiter paces outgoing requests to at
    most `settings.gemini_max_requests_per_minute` per rolling 60s
    window, so bursts of chunk/comparison calls don't trigger 429s in
    the first place. This is the primary defense; retries are the
    fallback for when the limit is still hit.
  - On a 429, the server's own `RetryInfo.retryDelay` is honored when
    present (more accurate than blind exponential backoff), otherwise
    we fall back to exponential backoff with jitter, up to
    `settings.gemini_max_retries` attempts.
  - If the 429 reports a *daily* quota violation (`QuotaFailure` with a
    `PerDay` quotaId - this is what Gemini's free tier actually returns:
    a handful of requests/day, not just requests/minute), retrying
    within the same run cannot succeed until the quota resets. Rather
    than burning through every remaining chunk/comparison issuing a
    doomed call each (observed: ~4 minutes of failing calls on a single
    27-page PDF), a process-wide circuit breaker trips immediately and
    every subsequent call fails fast for a cooldown window instead of
    hitting the API again.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

from google import genai
from google.genai import errors, types

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("llm_client")

_client: genai.Client | None = None


class LLMError(Exception):
    pass


class LLMNotConfiguredError(LLMError):
    pass


class LLMRateLimitedError(LLMError):
    """Raised when the Gemini API is rate-limiting us (429) and retries were
    either exhausted or futile (a daily quota, not a transient burst)."""


def _get_client() -> genai.Client:
    global _client
    if not settings.gemini_api_key:
        raise LLMNotConfiguredError("GEMINI_API_KEY is not set - see .env.example")
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


class _RateLimiter:
    """Paces calls to at most N per rolling 60s window and short-circuits
    everything during a daily-quota cooldown. One instance is shared by the
    whole process (including concurrent extraction workers), so it is the
    actual point of enforcement - not just each caller's own pacing."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._call_times: deque[float] = deque()
        self._quota_exhausted_until: float = 0.0

    def wait_for_slot(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                cooldown_remaining = self._quota_exhausted_until - now
                if cooldown_remaining > 0:
                    sleep_for = cooldown_remaining
                else:
                    rpm = max(settings.gemini_max_requests_per_minute, 1)
                    while self._call_times and now - self._call_times[0] >= 60.0:
                        self._call_times.popleft()
                    if len(self._call_times) < rpm:
                        self._call_times.append(now)
                        return
                    sleep_for = 60.0 - (now - self._call_times[0]) + 0.05
            time.sleep(max(sleep_for, 0.05))

    def mark_daily_quota_exhausted(self, cooldown_seconds: float) -> None:
        with self._lock:
            self._quota_exhausted_until = max(
                self._quota_exhausted_until, time.monotonic() + max(cooldown_seconds, 1.0)
            )
        logger.warning("Gemini daily quota exhausted - pausing all LLM calls for %.0fs", cooldown_seconds)

    def cooldown_remaining(self) -> float:
        with self._lock:
            return max(0.0, self._quota_exhausted_until - time.monotonic())


_rate_limiter = _RateLimiter()


def _parse_quota_error(exc: errors.APIError) -> tuple[float | None, bool]:
    """Returns (retry_delay_seconds, is_daily_quota) from a Gemini 429's
    error detail payload (RetryInfo / QuotaFailure), or (None, False) if the
    payload doesn't carry that structure."""
    details = getattr(exc, "details", None)
    if not isinstance(details, dict):
        return None, False

    delay: float | None = None
    is_daily = False
    for item in details.get("details", []) or []:
        item_type = item.get("@type", "")
        if item_type.endswith("RetryInfo"):
            raw = item.get("retryDelay", "")
            if isinstance(raw, str) and raw.endswith("s"):
                try:
                    delay = float(raw[:-1])
                except ValueError:
                    pass
        elif item_type.endswith("QuotaFailure"):
            for violation in item.get("violations", []) or []:
                if "PerDay" in (violation.get("quotaId") or ""):
                    is_daily = True
    return delay, is_daily


def _generate_with_retry(
    client: genai.Client, contents: list[types.Content], config: types.GenerateContentConfig
):
    cooldown = _rate_limiter.cooldown_remaining()
    if cooldown > 0:
        raise LLMRateLimitedError(
            f"Gemini daily quota exhausted; cooling down for another {cooldown:.0f}s before retrying."
        )

    last_exc: errors.ClientError | None = None
    for attempt in range(settings.gemini_max_retries + 1):
        _rate_limiter.wait_for_slot()
        try:
            return client.models.generate_content(model=settings.gemini_model, contents=contents, config=config)
        except errors.ClientError as exc:
            if exc.code != 429:
                raise
            last_exc = exc
            delay, is_daily = _parse_quota_error(exc)

            if is_daily:
                cooldown_seconds = delay or settings.gemini_daily_quota_cooldown_seconds
                _rate_limiter.mark_daily_quota_exhausted(cooldown_seconds)
                raise LLMRateLimitedError(
                    f"Gemini daily request quota exhausted (429 RESOURCE_EXHAUSTED); "
                    f"backing off {cooldown_seconds:.0f}s before further calls."
                ) from exc

            if attempt >= settings.gemini_max_retries:
                break

            backoff = delay if delay is not None else settings.gemini_retry_base_delay_seconds * (2**attempt)
            backoff += random.uniform(0, 0.5)
            logger.warning(
                "Gemini rate limit hit (attempt %d/%d) - retrying in %.1fs",
                attempt + 1,
                settings.gemini_max_retries,
                backoff,
            )
            time.sleep(backoff)

    raise LLMRateLimitedError(
        f"Gemini rate limit exceeded after {settings.gemini_max_retries} retries"
    ) from last_exc


@dataclass
class LLMJsonResult:
    ok: bool
    data: dict | list | None
    raw_text: str
    error: str | None = None
    rate_limited: bool = False


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
    Rate limiting (pacing + backoff + daily-quota circuit breaker) is
    handled transparently underneath - see module docstring.
    """
    client = _get_client()
    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)])
    ]
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=temperature,
        max_output_tokens=max_tokens,
        response_mime_type="application/json",
    )

    for attempt in range(2):
        try:
            response = _generate_with_retry(client, contents, config)
        except LLMRateLimitedError as exc:
            logger.error("Gemini rate limit: %s", exc)
            return LLMJsonResult(ok=False, data=None, raw_text="", error=str(exc), rate_limited=True)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM API call failed: %s", exc)
            return LLMJsonResult(ok=False, data=None, raw_text="", error=str(exc))

        raw_text = response.text or ""
        if not raw_text:
            candidates = getattr(response, "candidates", None) or []
            finish_reason = candidates[0].finish_reason if candidates else "unknown"
            logger.warning("LLM returned no text (finish_reason=%s)", finish_reason)
            return LLMJsonResult(
                ok=False, data=None, raw_text="", error=f"empty response (finish_reason={finish_reason})"
            )

        try:
            candidate = _extract_json_block(raw_text)
            data = json.loads(candidate)
            return LLMJsonResult(ok=True, data=data, raw_text=raw_text)
        except json.JSONDecodeError as exc:
            logger.warning("LLM returned invalid JSON (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                contents.append(types.Content(role="model", parts=[types.Part(text=raw_text)]))
                contents.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=(
                                    "Your previous response was not valid JSON "
                                    f"(error: {exc}). Reply again with ONLY valid JSON, "
                                    "no markdown fences, no commentary."
                                )
                            )
                        ],
                    )
                )
                continue
            return LLMJsonResult(ok=False, data=None, raw_text=raw_text, error=f"invalid JSON: {exc}")

    return LLMJsonResult(ok=False, data=None, raw_text="", error="exhausted retries")
