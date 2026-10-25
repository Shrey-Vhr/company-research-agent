from langchain_core.documents import Document

from research_agent.rag.chunker import chunk_documents

CHUNK_SIZE = 200
CHUNK_OVERLAP = 50


def make_page(company, pdf_page, n_words, fy="FY2025-26"):
    """Fake page. Every word starts with a tag unique to (company, page), e.g. 'tcs5w012'."""
    tag = f"{company.lower()}{pdf_page}"
    text = " ".join(f"{tag}w{i:03d}" for i in range(n_words))
    return Document(
        page_content=text,
        metadata={
            "company": company,
            "fy": fy,
            "source": f"{company.lower()}_fy2026.pdf",
            "pdf_page": pdf_page,
            "total_pages": 10,
        },
    )


def make_docs():
    """Three long pages (several chunks each) and one short page (one chunk), in loader order."""
    return [
        make_page("TCS", 1, 60),
        make_page("TCS", 2, 5),   # short: about 40 chars
        make_page("TCS", 5, 60),
        make_page("INFY", 5, 60),  # same pdf_page number as TCS p.5
    ]


def chunks_of(chunks, company, pdf_page):
    return [c for c in chunks if c.metadata["company"] == company and c.metadata["pdf_page"] == pdf_page]


def shared_edge(first, second):
    """Length of the longest end of `first` that is also the start of `second`."""
    for length in range(min(len(first), len(second)), 0, -1):
        if first.endswith(second[:length]):
            return length
    return 0


def test_no_chunk_longer_than_chunk_size():
    chunks = chunk_documents(make_docs(), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    assert len(chunks) > 4  # the long pages really were split
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)


def test_neighbouring_chunks_on_a_page_overlap():
    chunks = chunk_documents(make_docs(), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    page = chunks_of(chunks, "TCS", 1)
    assert len(page) >= 3
    for first, second in zip(page, page[1:]):
        edge = shared_edge(first.page_content, second.page_content)
        assert 0 < edge <= CHUNK_OVERLAP


def test_chunks_keep_their_pages_metadata():
    docs = make_docs()
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    originals = {(d.metadata["company"], d.metadata["pdf_page"]): d.metadata for d in docs}
    for c in chunks:
        original = originals[(c.metadata["company"], c.metadata["pdf_page"])]
        for key in ("company", "fy", "source", "pdf_page"):
            assert c.metadata[key] == original[key]


def test_chunks_never_mix_pages():
    chunks = chunk_documents(make_docs(), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for c in chunks:
        tag = f"{c.metadata['company'].lower()}{c.metadata['pdf_page']}"
        assert all(word.startswith(tag + "w") for word in c.page_content.split())


def test_chunk_number_starts_at_1_and_counts_up_per_page():
    chunks = chunk_documents(make_docs(), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for company, pdf_page in [("TCS", 1), ("TCS", 2), ("TCS", 5), ("INFY", 5)]:
        numbers = [c.metadata["chunk"] for c in chunks_of(chunks, company, pdf_page)]
        assert numbers == list(range(1, len(numbers) + 1))
    assert len(chunks_of(chunks, "TCS", 1)) > 1  # counting up was really tested


def test_same_pdf_page_in_two_companies_restarts_at_1():
    chunks = chunk_documents(make_docs(), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    tcs = chunks_of(chunks, "TCS", 5)
    infy = chunks_of(chunks, "INFY", 5)
    assert len(tcs) > 1 and len(infy) > 1
    assert infy[0].metadata["chunk"] == 1
    assert [c.metadata["chunk"] for c in infy] == list(range(1, len(infy) + 1))
    ids = [(c.metadata["company"], c.metadata["pdf_page"], c.metadata["chunk"]) for c in chunks]
    assert len(ids) == len(set(ids))  # no two chunks share a company/page/chunk ID


def test_short_page_gives_exactly_one_chunk():
    chunks = chunk_documents(make_docs(), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    page = chunks_of(chunks, "TCS", 2)
    assert len(page) == 1
    assert page[0].metadata["chunk"] == 1
    assert page[0].page_content == make_docs()[1].page_content


def test_no_chunk_is_empty_or_whitespace():
    chunks = chunk_documents(make_docs(), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    assert chunks
    assert all(c.page_content.strip() for c in chunks)
