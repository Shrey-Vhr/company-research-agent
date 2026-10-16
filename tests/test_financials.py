import json
from pathlib import Path

import pandas as pd
import pytest

from research_agent.tools import financials

FIXTURE = Path(__file__).parent / "fixtures" / "tcs_sample.json"


def load_statement(rows):
    """{row: {date: value}} -> DataFrame with rows as index and Timestamp columns, like yfinance."""
    df = pd.DataFrame(rows).T
    df.columns = pd.to_datetime(df.columns)
    return df


class FakeTicker:
    """Stands in for yf.Ticker so no test touches the internet."""

    def __init__(self, income, balance, currency):
        self.income_stmt = income
        self.balance_sheet = balance
        self.info = {"financialCurrency": currency}


@pytest.fixture
def sample():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {
        "income": load_statement(data["income_stmt"]),
        "balance": load_statement(data["balance_sheet"]),
        "currency": data["currency"],
    }


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    """Every test: cache goes to a temp folder, and any real yfinance call fails loudly."""
    monkeypatch.setattr(financials, "CACHE_DIR", tmp_path / "cache")

    def no_network(symbol):
        raise AssertionError(f"Test tried to call yfinance for {symbol}")

    monkeypatch.setattr(financials.yf, "Ticker", no_network)


def use_sample(monkeypatch, income, balance, currency="INR"):
    monkeypatch.setattr(financials.yf, "Ticker", lambda symbol: FakeTicker(income, balance, currency))


def test_parses_sample_into_crore(monkeypatch, sample):
    use_sample(monkeypatch, sample["income"], sample["balance"])

    result = financials.fetch_financials("tcs", years=2)

    assert result["ticker"] == "TCS.NS"
    assert result["unit"] == "INR crore"
    assert result["source"] == "Yahoo Finance via yfinance"
    assert [y["fiscal_year"] for y in result["years"]] == ["FY2024-25", "FY2023-24"]
    assert result["years"][0] == {
        "fiscal_year": "FY2024-25",
        "period_end": "2025-03-31",
        "revenue": 255324.0,
        "operating_income": 62293.0,
        "net_income": 48553.0,
        "total_debt": 9392.0,
        "total_equity": 94756.0,
    }
    assert result["notes"] == []


def test_missing_row_returns_null_with_note(monkeypatch, sample):
    balance_without_debt = sample["balance"].drop(index="Total Debt")
    use_sample(monkeypatch, sample["income"], balance_without_debt)

    result = financials.fetch_financials("TCS", years=1)

    year = result["years"][0]
    assert year["total_debt"] is None
    assert year["revenue"] == 255324.0  # other fields still filled
    assert len(result["notes"]) == 1
    assert "FY2024-25" in result["notes"][0]
    assert "total_debt" in result["notes"][0]


@pytest.mark.parametrize("ticker", ["", "   ", None, "TCS; DROP TABLE", "__import__('os')", "TCS.XX"])
def test_invalid_ticker_returns_error_without_network(ticker):
    # The autouse `offline` fixture makes any yfinance call fail, so this also proves none happened.
    result = financials.fetch_financials(ticker)

    assert set(result) == {"error"}
    assert "Invalid ticker" in result["error"]
