"""Build the Chroma index: load reports -> chunk -> estimate cost -> ask -> embed and save.

Run from the project root: python -m scripts.build_index
"""

from research_agent.rag.chunker import chunk_documents
from research_agent.rag.loader import load_reports
from research_agent.rag.store import add_chunks

# US dollars per 1 million input tokens for your EMBEDDING_MODEL. Fill in from https://openai.com/api/pricing/
PRICE_PER_MILLION_TOKENS = None

CHARS_PER_TOKEN = 4  # rough rule of thumb for English text


if __name__ == "__main__":
    chunks = chunk_documents(load_reports())
    chars = sum(len(c.page_content) for c in chunks)
    tokens = chars / CHARS_PER_TOKEN
    print(f"\nChunks: {len(chunks)}")
    print(f"Characters: {chars:,} -> about {tokens:,.0f} tokens")
    if PRICE_PER_MILLION_TOKENS is None:
        print("Cost: unknown, fill in PRICE_PER_MILLION_TOKENS in scripts/build_index.py")
    else:
        print(f"Cost: about ${tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS:.4f}")

    if input("Embed and save these chunks? (y/n) ").strip().lower() != "y":
        print("Stopped. Nothing was embedded.")
        raise SystemExit(0)

    store = add_chunks(chunks)
    count = len(store.get(include=[])["ids"])  # include=[] fetches only the IDs, not texts or vectors
    print(f"Collection now holds {count} items")
