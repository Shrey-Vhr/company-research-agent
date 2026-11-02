"""Data helper: load the annual-report PDFs as one LangChain Document per page."""

from pathlib import Path

import pymupdf
from langchain_core.documents import Document

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"

# File name -> (company as NSE ticker, same as get_financials input, financial year). Must match docs/DATA_SOURCES.md.
REPORTS = {
    "tcs_fy2026.pdf": ("TCS", "FY2025-26"),
    "infy_fy2026.pdf": ("INFY", "FY2025-26"),
    "wipro_fy2026.pdf": ("WIPRO", "FY2025-26"),
}

# Pages with less text than this are skipped: blank covers, "www.infosys.com", a lone "Notes" heading.
MIN_CHARS = 50

# Page numbering (checked on all three PDFs):
# - pymupdf numbers pages from 0 (enumerate below).
# - The printed number on a page is not usable: the PDFs have no page labels, many PDF pages are
#   two-page spreads (TCS PDF page 21 shows printed pages 38 and 39), and the extracted text
#   sometimes splits digits ("1 31" for 131).
# So we cite pdf_page = index + 1, the number you type into a PDF viewer to check it.


def load_report(path, company, fy):
    """Return one Document per non-empty page of one PDF, with company, fy and pdf_page in metadata."""
    docs = []
    with pymupdf.open(path) as pdf:
        total_pages = len(pdf)
        for index, page in enumerate(pdf):  # index starts at 0
            text = page.get_text().strip()
            if len(text) < MIN_CHARS:
                continue
            docs.append(Document(
                page_content=text,
                # Only what we filter or cite on.
                metadata={
                    "company": company,
                    "fy": fy,
                    "source": Path(path).name,
                    "pdf_page": index + 1,  # 1-based PDF page; see note above
                    "total_pages": total_pages,
                },
            ))
    print(f"{Path(path).name}: kept {len(docs)} of {total_pages} pages "
          f"(skipped {total_pages - len(docs)} with < {MIN_CHARS} chars); citing pdf_page (1-based)")
    return docs


def load_reports(reports_dir=REPORTS_DIR):
    """Load every report listed in REPORTS. Raises FileNotFoundError if one is missing."""
    docs = []
    for file_name, (company, fy) in REPORTS.items():
        path = Path(reports_dir) / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing report {path}; see docs/DATA_SOURCES.md")
        docs.extend(load_report(path, company, fy))
    return docs


if __name__ == "__main__":
    docs = load_reports()
    print(f"Total: {len(docs)} pages")
    # Print a few pages from each report to eyeball the text and metadata.
    for company in ("TCS", "INFY", "WIPRO"):
        company_docs = [d for d in docs if d.metadata["company"] == company]
        for doc in (company_docs[0], company_docs[len(company_docs) // 2]):
            print("\n" + "=" * 80)
            print(doc.metadata)
            print(doc.page_content[:600])
