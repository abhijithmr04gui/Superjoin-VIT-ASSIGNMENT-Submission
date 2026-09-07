from app.pipeline.normalization import normalize_date, normalize_entity_name, normalize_number


def test_normalize_number_dollar_million():
    r = normalize_number("$10 million")
    assert r.parse_ok
    assert r.value == 10_000_000
    assert r.currency == "USD"


def test_normalize_number_usd_m_abbreviation():
    r = normalize_number("USD 10M")
    assert r.parse_ok
    assert r.value == 10_000_000
    assert r.currency == "USD"


def test_normalize_number_comma_dollars():
    r = normalize_number("10,000,000 dollars")
    assert r.parse_ok
    assert r.value == 10_000_000
    assert r.currency == "USD"


def test_normalize_number_crore_inr():
    r = normalize_number("₹4.2 crore")
    assert r.parse_ok
    assert r.currency == "INR"
    assert abs(r.value - 42_000_000) < 1


def test_normalize_number_percent():
    r = normalize_number("15%")
    assert r.parse_ok
    assert r.value == 15
    assert r.unit == "percent"


def test_normalize_number_no_numeric_token():
    r = normalize_number("a lot of money")
    assert not r.parse_ok


def test_normalize_date_fiscal_year():
    d = normalize_date("FY2024")
    assert d.parse_ok
    assert d.precision == "period"
    assert d.year == 2024


def test_normalize_date_quarter():
    d = normalize_date("Q3 2024")
    assert d.parse_ok
    assert d.year == 2024
    assert d.month == 7


def test_normalize_date_month_year():
    d = normalize_date("January 2025")
    assert d.parse_ok
    assert d.year == 2025
    assert d.month == 1


def test_normalize_date_empty():
    d = normalize_date("")
    assert not d.parse_ok


def test_normalize_entity_name_strips_suffixes():
    assert normalize_entity_name("ABC Technologies Ltd.") == normalize_entity_name("ABC Technologies")
    assert normalize_entity_name("ABC Tech Ltd") != normalize_entity_name("ABC Technologies")
