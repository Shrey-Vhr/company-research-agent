import re

# Things that look like numbers but are labels, removed before extracting.
# ISO dates and times (2026-03-31, 2026-10-26T11:02:00.003372+00:00, 11:02:00) go first,
# so the FY pattern below cannot half-match a date.
_TIME = r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"
ISO_DATETIME = re.compile(rf"\b\d{{4}}-\d{{2}}-\d{{2}}(?:[T ]{_TIME})?|\b{_TIME}")
FY_LABEL = re.compile(r"\bFY\s?\d{4}(?:\s?[-–/]\s?\d{2,4})?|\b\d{4}\s?[-–]\s?\d{2,4}\b", re.IGNORECASE)
PAGE_CITATION = re.compile(r"\b(?:pp?\.|pages?)\s?\d+(?:\s?[-–]\s?\d+)?", re.IGNORECASE)

# Optional minus, digits with optional comma groups (5,84,519 or 584,519), optional decimals.
NUMBER = re.compile(r"(?<!\w)-?\d+(?:,\d+)*(?:\.\d+)?")


def _clean(text):
    """Blank out dates, times, FY labels and page citations."""
    text = ISO_DATETIME.sub(" ", text)
    text = FY_LABEL.sub(" ", text)
    return PAGE_CITATION.sub(" ", text)


def _extract(text):
    """Return (value, decimal_places) for each number in text."""
    text = _clean(text)
    found = []
    for token in NUMBER.findall(text):
        token = token.replace(",", "")
        places = len(token.split(".")[1]) if "." in token else 0
        found.append((float(token), places))
    return found


def extract_numbers(text):
    return [value for value, _ in _extract(text)]


def find_unsupported(draft, evidence_texts):
    """Numbers in the draft that no evidence number equals or rounds to."""
    evidence_numbers = [n for text in evidence_texts for n in extract_numbers(text)]
    unsupported = []
    for value, places in _extract(draft):
        supported = any(abs(round(e, places) - value) < 1e-9 for e in evidence_numbers)
        if not supported:
            unsupported.append(value)
    return unsupported


def _with_context(text, width=30):
    """Like _extract, but also return the text around each number."""
    text = _clean(text)
    found = []
    for m in NUMBER.finditer(text):
        token = m.group().replace(",", "")
        places = len(token.split(".")[1]) if "." in token else 0
        snippet = text[max(0, m.start() - width):m.end() + width].replace("\n", " ")
        found.append((float(token), places, snippet))
    return found


def explain_support(draft, evidence_texts):
    """Debug: print which evidence number (if any) supports each draft number."""
    evidence = [item for text in evidence_texts for item in _with_context(text)]
    for value, places, snippet in _with_context(draft):
        print(f"draft {value}  ...{snippet}...")
        match = next(((e, s) for e, _, s in evidence if abs(round(e, places) - value) < 1e-9), None)
        if match is None:
            print("   -> UNSUPPORTED")
        else:
            print(f"   -> supported by evidence {match[0]}  ...{match[1]}...")
