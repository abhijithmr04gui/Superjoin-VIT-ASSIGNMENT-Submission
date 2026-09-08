"""
Unit tests for the rate limit management in app/pipeline/llm_client.py:
request pacing, retry-with-backoff honoring the server's own retryDelay,
and the daily-quota circuit breaker. These exercise _generate_with_retry
directly against a fake client so no network/API key is needed - the
same reason the pipeline integration test mocks call_json itself rather
than going through here.
"""
from __future__ import annotations

import time

import pytest
from google.genai import errors

from app.core.config import settings
from app.pipeline import llm_client
from app.pipeline.llm_client import LLMRateLimitedError, _RateLimiter, _generate_with_retry, _parse_quota_error


def _make_429(retry_delay: str | None, quota_id: str) -> errors.ClientError:
    details: list[dict] = []
    if retry_delay is not None:
        details.append({"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": retry_delay})
    details.append(
        {
            "@type": "type.googleapis.com/google.rpc.QuotaFailure",
            "violations": [{"quotaId": quota_id}],
        }
    )
    payload = {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota exceeded", "details": details}
    return errors.ClientError(429, payload, response=None)


PER_MINUTE_QUOTA = "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"
PER_DAY_QUOTA = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.calls = 0

    class _Models:
        def __init__(self, outer: "_FakeClient") -> None:
            self._outer = outer

        def generate_content(self, model, contents, config):
            self._outer.calls += 1
            item = self._outer._responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    @property
    def models(self):
        return self._Models(self)


@pytest.fixture(autouse=True)
def _fresh_rate_limiter(monkeypatch):
    # The rate limiter is a process-wide singleton by design (it must be,
    # to actually gate concurrent extraction workers) - give each test its
    # own instance so they don't see each other's cooldowns/call history.
    monkeypatch.setattr(llm_client, "_rate_limiter", _RateLimiter())
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(settings, "gemini_max_retries", 3)
    monkeypatch.setattr(settings, "gemini_retry_base_delay_seconds", 0.01)
    monkeypatch.setattr(settings, "gemini_max_requests_per_minute", 100)
    monkeypatch.setattr(settings, "gemini_daily_quota_cooldown_seconds", 60)
    yield


def test_parse_quota_error_extracts_retry_delay_and_daily_flag():
    exc = _make_429("51s", PER_DAY_QUOTA)
    delay, is_daily = _parse_quota_error(exc)
    assert delay == 51.0
    assert is_daily is True


def test_parse_quota_error_per_minute_is_not_daily():
    exc = _make_429("2.5s", PER_MINUTE_QUOTA)
    delay, is_daily = _parse_quota_error(exc)
    assert delay == 2.5
    assert is_daily is False


def test_parse_quota_error_missing_retry_info_returns_none_delay():
    payload = {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota exceeded", "details": []}
    exc = errors.ClientError(429, payload, response=None)
    delay, is_daily = _parse_quota_error(exc)
    assert delay is None
    assert is_daily is False


def test_transient_rate_limit_retries_then_succeeds():
    client = _FakeClient([_make_429("0.01s", PER_MINUTE_QUOTA), "second-call-response"])
    result = _generate_with_retry(client, contents=[], config=object())
    assert result == "second-call-response"
    assert client.calls == 2


def test_transient_rate_limit_gives_up_after_max_retries():
    # max_retries=3 -> 4 total attempts, all 429s -> exhausted
    responses = [_make_429("0.01s", PER_MINUTE_QUOTA) for _ in range(4)]
    client = _FakeClient(responses)
    with pytest.raises(LLMRateLimitedError):
        _generate_with_retry(client, contents=[], config=object())
    assert client.calls == 4


def test_non_rate_limit_client_error_is_not_retried():
    payload = {"code": 400, "status": "INVALID_ARGUMENT", "message": "bad request", "details": []}
    client = _FakeClient([errors.ClientError(400, payload, response=None)])
    with pytest.raises(errors.ClientError):
        _generate_with_retry(client, contents=[], config=object())
    assert client.calls == 1


def test_daily_quota_trips_breaker_and_short_circuits_subsequent_calls():
    client = _FakeClient([_make_429("51s", PER_DAY_QUOTA)])

    with pytest.raises(LLMRateLimitedError):
        _generate_with_retry(client, contents=[], config=object())
    assert client.calls == 1

    # Breaker is now tripped - a second call must fail fast without ever
    # touching the client again, instead of burning through every
    # remaining chunk/comparison with a doomed request each.
    with pytest.raises(LLMRateLimitedError):
        _generate_with_retry(client, contents=[], config=object())
    assert client.calls == 1
    assert llm_client._rate_limiter.cooldown_remaining() > 0


def test_rate_limiter_caps_calls_within_rolling_window(monkeypatch):
    limiter = _RateLimiter()
    monkeypatch.setattr(settings, "gemini_max_requests_per_minute", 2)

    fake_now = [0.0]
    sleeps: list[float] = []

    def fake_monotonic():
        return fake_now[0]

    def fake_sleep(seconds):
        sleeps.append(seconds)
        fake_now[0] += seconds

    monkeypatch.setattr(time, "monotonic", fake_monotonic)
    monkeypatch.setattr(time, "sleep", fake_sleep)

    limiter.wait_for_slot()
    limiter.wait_for_slot()
    assert sleeps == []  # first two calls fit inside the limit for free

    limiter.wait_for_slot()  # third call within the same window must wait
    assert sum(sleeps) > 0
