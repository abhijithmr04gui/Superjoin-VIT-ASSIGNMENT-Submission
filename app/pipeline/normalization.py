"""
Deterministic normalization. No LLM calls in this module by design
(assignment section 11 / 16: things code can do reliably should not be
delegated to the LLM).

Every function here is pure and unit-testable in isolation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from dateutil import parser as dateparser

_MAGNITUDE = {
    "thousand": 1_000,
    "k": 1_000,
    "million": 1_000_000,
    "mn": 1_000_000,
    "m": 1_000_000,
    "billion": 1_000_000_000,
    "bn": 1_000_000_000,
    "b": 1_000_000_000,
    "crore": 10_000_000,
    "lakh": 100_000,
}

_CURRENCY_SYMBOLS = {
    "$": "USD",
    "₹": "INR",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
}

_CURRENCY_WORDS = {
    "usd": "USD",
    "dollars": "USD",
    "dollar": "USD",
    "us dollars": "USD",
    "inr": "INR",
    "rupees": "INR",
    "rs": "INR",
    "rs.": "INR",
    "eur": "EUR",
    "euros": "EUR",
    "gbp": "GBP",
    "pounds": "GBP",
    "jpy": "JPY",
    "yen": "JPY",
}


@dataclass
class NormalizedNumber:
    value: float | None
    currency: str | None = None
    unit: str | None = None  # e.g. "percent", "count" - non-currency units
    raw: str = ""
    parse_ok: bool = True
    notes: str = ""


def normalize_number(raw_text: str) -> NormalizedNumber:
    """
    Parse strings like:
      "$10 million", "USD 10M", "10,000,000 dollars", "₹4.2 crore",
      "15%", "12.4 million dollars"
    into a normalized float plus currency/unit.
    """
    text = raw_text.strip()
    if not text:
        return NormalizedNumber(value=None, raw=raw_text, parse_ok=False, notes="empty input")

    lowered = text.lower()

    currency = None
    for sym, code in _CURRENCY_SYMBOLS.items():
        if sym in text:
            currency = code
            break
    if currency is None:
        for word, code in _CURRENCY_WORDS.items():
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                currency = code
                break

    is_percent = "%" in text or "percent" in lowered

    # Find the numeric token together with an optional magnitude suffix
    # directly attached to it (e.g. "10M", "4.2bn") or following as a
    # separate word (e.g. "10 million"). Attached suffixes need their own
    # pass since "\bm\b" never matches inside "10m" (no boundary between
    # a digit and a letter).
    combined_match = re.search(
        r"[-+]?\d[\d,]*\.?\d*\s*(thousand|million|billion|crore|lakh|mn|bn|k|m|b)?",
        lowered,
    )
    num_only_match = re.search(r"[-+]?\d[\d,]*\.?\d*", text)
    if not num_only_match:
        return NormalizedNumber(
            value=None, currency=currency, raw=raw_text, parse_ok=False, notes="no numeric token found"
        )

    num_str = num_only_match.group(0).replace(",", "")
    try:
        value = float(num_str)
    except ValueError:
        return NormalizedNumber(
            value=None, currency=currency, raw=raw_text, parse_ok=False, notes="failed float conversion"
        )

    magnitude_applied = None
    attached_suffix = combined_match.group(1) if combined_match else None
    if attached_suffix and attached_suffix in _MAGNITUDE:
        value *= _MAGNITUDE[attached_suffix]
        magnitude_applied = attached_suffix
    else:
        # Fall back to a standalone magnitude word anywhere in the text
        # (e.g. currency and number separated: "$10 million total").
        for word, mult in sorted(_MAGNITUDE.items(), key=lambda kv: -len(kv[0])):
            if len(word) > 1:
                pattern = rf"\b{re.escape(word)}\b"
            else:
                # single-letter word (k/m/b): only match if NOT glued to a digit
                # (that case is handled by attached_suffix above already).
                pattern = rf"(?<![\d.]){re.escape(word)}\b"
            if re.search(pattern, lowered):
                value *= mult
                magnitude_applied = word
                break

    unit = "percent" if is_percent else None

    return NormalizedNumber(
        value=value,
        currency=currency,
        unit=unit,
        raw=raw_text,
        parse_ok=True,
        notes=f"magnitude={magnitude_applied}" if magnitude_applied else "",
    )


@dataclass
class NormalizedDate:
    iso_date: str | None  # "2025-01-15" if fully resolvable
    year: int | None = None
    month: int | None = None
    day: int | None = None
    precision: str = "unknown"  # day/month/year/period/unknown
    raw: str = ""
    parse_ok: bool = True
    notes: str = ""


_FISCAL_PERIOD_RE = re.compile(r"\bFY\s?['’]?(\d{2,4})\b", re.IGNORECASE)
_QUARTER_RE = re.compile(r"\bQ([1-4])\s?['’]?(\d{2,4})\b", re.IGNORECASE)


def normalize_date(raw_text: str) -> NormalizedDate:
    """
    Handles explicit dates ("January 2025", "2025-01", "Jan 15 2025") as
    well as reporting-period style strings ("FY2024", "Q3 2024") which are
    not full calendar dates but still need a comparable normalized form.
    """
    text = raw_text.strip()
    if not text:
        return NormalizedDate(iso_date=None, raw=raw_text, parse_ok=False, notes="empty input")

    fy_match = _FISCAL_PERIOD_RE.search(text)
    if fy_match:
        year_str = fy_match.group(1)
        year = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)
        return NormalizedDate(
            iso_date=f"{year}-01-01",
            year=year,
            precision="period",
            raw=raw_text,
            parse_ok=True,
            notes=f"fiscal_year={year}",
        )

    q_match = _QUARTER_RE.search(text)
    if q_match:
        quarter = int(q_match.group(1))
        year_str = q_match.group(2)
        year = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)
        month = (quarter - 1) * 3 + 1
        return NormalizedDate(
            iso_date=f"{year}-{month:02d}-01",
            year=year,
            month=month,
            precision="period",
            raw=raw_text,
            parse_ok=True,
            notes=f"quarter=Q{quarter}",
        )

    try:
        default_dt = datetime(1, 1, 1)
        parsed = dateparser.parse(text, default=default_dt, fuzzy=True)
    except (ValueError, OverflowError):
        return NormalizedDate(iso_date=None, raw=raw_text, parse_ok=False, notes="dateutil could not parse")

    if parsed is None:
        return NormalizedDate(iso_date=None, raw=raw_text, parse_ok=False, notes="dateutil returned None")

    has_day = bool(re.search(r"\b\d{1,2}\b", text)) and parsed.day != 1
    has_month = bool(re.search(r"[A-Za-z]{3,9}|/|-", text))

    if has_day:
        precision = "day"
    elif has_month:
        precision = "month"
    else:
        precision = "year"

    return NormalizedDate(
        iso_date=parsed.date().isoformat(),
        year=parsed.year,
        month=parsed.month if precision != "year" else None,
        day=parsed.day if precision == "day" else None,
        precision=precision,
        raw=raw_text,
        parse_ok=True,
    )


def normalize_entity_name(raw_name: str) -> str:
    """
    Lightweight deterministic canonicalization used as a first pass before
    (optional) LLM-assisted entity resolution: lowercase, strip common
    legal suffixes and punctuation, collapse whitespace.
    """
    name = raw_name.strip().lower()
    name = re.sub(r"[.,]", "", name)
    suffixes = [
        r"\bltd\b", r"\blimited\b", r"\binc\b", r"\bincorporated\b",
        r"\bcorp\b", r"\bcorporation\b", r"\bllc\b", r"\bllp\b",
        r"\bplc\b", r"\bco\b", r"\bcompany\b",
    ]
    for suf in suffixes:
        name = re.sub(suf, "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


@dataclass
class NormalizationResult:
    normalized_value: dict = field(default_factory=dict)
    unit: str | None = None
    currency: str | None = None
    date: str | None = None
    reporting_period: str | None = None
    notes: str = ""
