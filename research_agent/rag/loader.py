"""Data helper: load the annual-report PDFs as one LangChain Document per page."""

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader

REPORTS_DIR = Path(__file__).resolve().parents[2] / "data" / "reports"

# File name -> (company, financial year). Must match docs/DATA_SOURCES.md.
REPORTS = {
    "tcs_fy2026.pdf": ("TCS", "FY2025-26"),
    "infy_fy2026.pdf": ("Infosys", "FY2025-26"),
    "wipro_fy2026.pdf": ("Wipro", "FY2025-26"),
}

# Pages with less text than this are skipped: blank covers, "www.infosys.com", a lone "Notes" heading.
MIN_CHARS = 50

# Page numbering (checked on all three PDFs):
# - PyMuPDFLoader's metadata["page"] starts at 0.
# - The printed number on a page is not usable: the PDFs have no page labels, many PDF pages are
#   two-page spreads (TCS PDF page 21 shows printed pages 38 and 39), and the extracted text
#   sometimes splits digits ("1 31" for 131).
# So we cite pdf_page = metadata["page"] + 1, the number you type into a PDF viewer to check it.


def load_report(path, company, fy):
    """Return one Document per non-empty page of one PDF, with company, fy and pdf_page in metadata."""
    pages = PyMuPDFLoader(str(path)).load()
    docs = []
    for page in pages:
        text = page.page_content.strip()
        if len(text) < MIN_CHARS:
            continue
        page.page_content = text
        # Replace the loader's metadata (creator, producer, dates...) with only what we filter or cite on.
        page.metadata = {
            "company": company,
            "fy": fy,
            "source": Path(path).name,
            "pdf_page": page.metadata["page"] + 1,  # 1-based PDF page; see note above
            "total_pages": page.metadata["total_pages"],
        }
        docs.append(page)
    print(f"{Path(path).name}: kept {len(docs)} of {len(pages)} pages "
          f"(skipped {len(pages) - len(docs)} with < {MIN_CHARS} chars); citing pdf_page (1-based)")
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
    for company in ("TCS", "Infosys", "Wipro"):
        company_docs = [d for d in docs if d.metadata["company"] == company]
        for doc in (company_docs[0], company_docs[len(company_docs) // 2]):
            print("\n" + "=" * 80)
            print(doc.metadata)
            print(doc.page_content[:600])
