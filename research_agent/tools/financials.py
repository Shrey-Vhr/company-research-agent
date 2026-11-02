"""Data helper: yearly financials for an NSE company from Yahoo Finance (via yfinance)."""

import json
import math
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from langchain.tools import tool

SOURCE = "Yahoo Finance via yfinance"
CRORE = 10_000_000  # 1 crore = 10^7 rupees
MAX_YEARS = 4  # yfinance only keeps about 4 years of annual statements

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"
CACHE_MAX_AGE = timedelta(days=7)
CACHE_VERSION = 1  # bump when the saved format changes, so old files are re-downloaded

# Letters, digits, & and - (e.g. "M&M", "BAJAJ-AUTO"), optionally ending in .NS or .BO
TICKER_PATTERN = re.compile(r"^[A-Z0-9&-]{1,20}(\.(NS|BO))?$")

COVERED = {"TCS", "INFY", "WIPRO"}
# Output field -> (statement, yfinance row names to try in order)
FIELDS = {
    "revenue": ("income", ["Total Revenue", "Operating Revenue"]),
    "operating_income": ("income", ["Operating Income"]),
    "net_income": ("income", ["Net Income", "Net Income Common Stockholders"]),
    "total_debt": ("balance", ["Total Debt"]),
    "total_equity": ("balance", ["Stockholders Equity", "Common Stock Equity"]),
}


def normalize_ticker(ticker: object) -> str | None:
    """Return an upper-case ticker ending in .NS/.BO, or None if it looks wrong."""
    if not isinstance(ticker, str):
        return None
    ticker = ticker.strip().upper()
    if not TICKER_PATTERN.match(ticker):
        return None
    if not ticker.endswith((".NS", ".BO")):
        ticker += ".NS"
    return ticker


def fiscal_year_label(period_end: pd.Timestamp) -> str:
    """2025-03-31 -> 'FY2024-25' (Indian financial year ends in March)."""
    end_year = period_end.year if period_end.month <= 3 else period_end.year + 1
    return f"FY{end_year - 1}-{str(end_year)[2:]}"


def average_fx_to_inr(currency: str, period_end: pd.Timestamp) -> float | None:
    """Average daily rate (1 unit of currency -> INR) over the year ending period_end, or None."""
    start = period_end - pd.DateOffset(years=1) + pd.DateOffset(days=1)
    try:
        history = yf.Ticker(f"{currency}INR=X").history(
            start=start, end=period_end + pd.DateOffset(days=1)
        )
    except Exception:  # noqa: BLE001 - yfinance raises many error types; never crash
        return None
    if history.empty:
        return None
    return float(history["Close"].mean())


def get_value_in_crore(
    statement: pd.DataFrame,
    row_names: list[str],
    column: pd.Timestamp,
    fx_rate: float = 1.0,
) -> float | None:
    """Return value in ₹ crore, or None if the row or value is missing."""
    if column not in statement.columns:
        return None
    for row in row_names:
        if row in statement.index:
            value = statement.loc[row, column]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            return round(float(value) * fx_rate / CRORE, 2)
    return None


def download_financials(symbol: str) -> dict:
    """Download MAX_YEARS of data for an already-normalized symbol. Returns a dict or {"error": ...}."""
    try:
        t = yf.Ticker(symbol)
        income = t.income_stmt
        balance = t.balance_sheet
    except Exception as e:  # noqa: BLE001 - network failure, timeout, Yahoo changing its API
        return {
            "error": f"Could not download data for {symbol}: {type(e).__name__}: {e}"
        }

    if income is None or income.empty:
        return {
            "error": f"No financial data found for {symbol}. "
            "The ticker may not exist, or the network/Yahoo request failed."
        }

    if balance is None:
        balance = income.iloc[
            0:0
        ]  # empty table, so balance-sheet fields become null with a note

    notes = []

    # Some NSE companies (e.g. Infosys) are stored by Yahoo in USD, not INR.
    try:
        currency = t.info.get("financialCurrency") or "INR"
    except Exception:  # noqa: BLE001 - yfinance raises many error types; never crash
        currency = "INR"
        notes.append("Could not confirm the reporting currency; assumed INR.")

    results = []
    for column in sorted(income.columns, reverse=True)[:MAX_YEARS]:  # newest year first
        year = {
            "fiscal_year": fiscal_year_label(column),
            "period_end": column.strftime("%Y-%m-%d"),
        }

        fx_rate = 1.0
        if currency != "INR":
            fx_rate = average_fx_to_inr(currency, column)
            if fx_rate is None:
                notes.append(
                    f"{year['fiscal_year']}: reported in {currency} and no {currency}/INR rate "
                    "was available, so all values for this year are null."
                )
            else:
                notes.append(
                    f"{year['fiscal_year']}: reported in {currency}; converted at the year's "
                    f"average rate of {fx_rate:.2f} INR per {currency}, so values are approximate."
                )

        for field, (which, row_names) in FIELDS.items():
            statement = income if which == "income" else balance
            value = (
                None
                if fx_rate is None
                else get_value_in_crore(statement, row_names, column, fx_rate)
            )
            year[field] = value
            if fx_rate is None:
                continue
            if value is None:
                notes.append(
                    f"{year['fiscal_year']}: {field} not available "
                    f"(looked for {', '.join(row_names)} in the {which} statement)."
                )
        results.append(year)

    return {
        "ticker": symbol,
        "unit": "INR crore",
        "years": results,
        "notes": notes,
        "source": SOURCE,
        "fetched_at": datetime.now(UTC).isoformat(),
        "cache_version": CACHE_VERSION,
    }


def cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.json"


def read_cache(symbol: str) -> dict | None:
    """Return the cached result if it exists, is readable and is under 7 days old; else None."""
    try:
        data = json.loads(cache_path(symbol).read_text(encoding="utf-8"))
        fetched_at = datetime.fromisoformat(data["fetched_at"])
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ):  # no file, bad JSON, missing/odd fetched_at
        return None
    if data.get("cache_version") != CACHE_VERSION:
        return None
    if datetime.now(UTC) - fetched_at > CACHE_MAX_AGE:
        return None
    return data


def write_cache(symbol: str, data: dict) -> None:
    """Save to data/cache/<symbol>.json. A failed write is ignored: the cache is only a speed-up."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cache_path(symbol).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(
            cache_path(symbol)
        )  # swap in whole file, so a crash never leaves half a JSON
    except OSError:
        pass


def fetch_financials(ticker: str, years: int = 3) -> dict:
    """Yearly revenue, operating income, net income, debt and equity in ₹ crore.

    Uses data/cache/<ticker>.json if it is under 7 days old, otherwise downloads and saves it.
    Returns a dict. On bad input or a failed download it returns {"error": "..."}
    instead of raising.
    """
    symbol = normalize_ticker(ticker)
    if symbol is None:
        return {
            "error": f"Invalid ticker {ticker!r}. Use an NSE symbol like 'TCS' or 'TCS.NS'."
        }
    if not isinstance(years, int) or not 1 <= years <= MAX_YEARS:
        return {
            "error": f"years must be a whole number from 1 to {MAX_YEARS} (yfinance keeps about 4 years)."
        }

    data = read_cache(symbol)
    from_cache = data is not None
    if data is None:
        data = download_financials(symbol)
        if "error" in data:
            return data  # errors are never cached, so the next call tries again
        write_cache(symbol, data)

    kept = data["years"][:years]
    kept_labels = {y["fiscal_year"] for y in kept}
    # Year notes start with "FY....:"; keep only those for the years returned, plus general notes.
    notes = [
        n
        for n in data["notes"]
        if not n.startswith("FY") or n.split(":")[0] in kept_labels
    ]

    return {
        "ticker": data["ticker"],
        "unit": data["unit"],
        "years": kept,
        "notes": notes,
        "source": data["source"],
        "fetched_at": data["fetched_at"],
        "from_cache": from_cache,
    }


@tool
def get_financials(ticker: str, years: int = 3) -> dict:
    """Get yearly financials for an Indian company listed on the NSE.(it covers only TCS, INFY and WIPRO.)

    Use this for any question about a company's revenue, operating income, net income,
    total debt or total equity. To compare companies, call it once per company.
    Do not use it for growth, margins or ratios: get the numbers here, then use `calculate`.
    It has no quarterly data, share prices or news.

    Args:
        ticker: NSE symbol, not the company name. Examples: "TCS", "INFY", "WIPRO". ".NS" is optional.
        years: how many financial years to return, 1 to 4, newest first. Default 3.

    All amounts are in ₹ crore (1 crore = 10,000,000 rupees).
    "FY2024-25" is the Indian financial year from 1 April 2024 to 31 March 2025.
    A null value means the figure is missing; never guess it. Always read "notes": some values
    are converted from USD and are approximate. If the result has an "error" key, the call failed;
    fix the ticker or tell the user, and do not make up numbers. Cite the "source" field.
    """
    symbol = ticker.strip().upper().removesuffix(".NS")
    if symbol not in COVERED:
        return {
            "error": f"{ticker} is not covered. Only TCS, INFY and WIPRO are available in this version."
        }
    return fetch_financials(ticker, years)
