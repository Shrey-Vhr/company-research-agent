"""Build the Chroma index: load reports -> chunk -> estimate cost -> ask -> embed and save.

Run from the project root: python -m scripts.build_index
Options: --chunk-size 1600 --chunk-overlap 200 --collection annual_reports_1600 (these are the defaults)
"""

import argparse

from research_agent.rag.chunker import chunk_documents
from research_agent.rag.loader import load_reports
from research_agent.rag.store import add_chunks

# US dollars per 1 million input tokens for your EMBEDDING_MODEL. Fill in from https://openai.com/api/pricing/
PRICE_PER_MILLION_TOKENS = 0.02

CHARS_PER_TOKEN = 4  # rough rule of thumb for English text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=1600)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--collection", default="annual_reports_1600")
    args = parser.parse_args()

    print(f"Chunk size {args.chunk_size}, overlap {args.chunk_overlap}, collection '{args.collection}'")
    chunks = chunk_documents(load_reports(), chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
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

    store = add_chunks(chunks, args.collection)
    count = len(store.get(include=[])["ids"])  # include=[] fetches only the IDs, not texts or vectors
    print(f"Collection now holds {count} items")
