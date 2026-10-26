import re

# Things that look like numbers but are labels, removed before extracting.
FY_LABEL = re.compile(r"\bFY\s?\d{4}(?:\s?[-–/]\s?\d{2,4})?|\b\d{4}\s?[-–]\s?\d{2,4}\b", re.IGNORECASE)
PAGE_CITATION = re.compile(r"\b(?:pp?\.|pages?)\s?\d+(?:\s?[-–]\s?\d+)?", re.IGNORECASE)

# Optional minus, digits with optional comma groups (5,84,519 or 584,519), optional decimals.
NUMBER = re.compile(r"(?<!\w)-?\d+(?:,\d+)*(?:\.\d+)?")


def _extract(text):
    """Return (value, decimal_places) for each number in text."""
    text = FY_LABEL.sub(" ", text)
    text = PAGE_CITATION.sub(" ", text)
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
