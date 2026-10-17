"""Data helper: the Chroma vector store that holds the annual-report chunks."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
COLLECTION = "annual_reports"
# Chunks per add_documents call. 100 chunks of up to 3,200 chars is about 80,000 tokens,
# well under OpenAI's 300,000-token limit for one embeddings request.
BATCH_SIZE = 100


def get_store():
    """Return the Chroma store saved in data/chroma, embedding with EMBEDDING_MODEL from .env."""
    embeddings = OpenAIEmbeddings(model=os.environ["EMBEDDING_MODEL"])
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def chunk_id(chunk):
    """Fixed ID from metadata, e.g. 'tcs-fy2025-26-p181-c2', so re-adding a chunk replaces it."""
    m = chunk.metadata
    return f"{m['company']}-{m['fy']}-p{m['pdf_page']}-c{m['chunk']}".lower()


def add_chunks(chunks):
    """Embed and save chunks in batches of BATCH_SIZE. Chroma upserts by ID, so nothing is duplicated."""
    store = get_store()
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        store.add_documents(batch, ids=[chunk_id(c) for c in batch])
        print(f"Added chunks {start + 1}-{start + len(batch)} of {len(chunks)}")
    return store
