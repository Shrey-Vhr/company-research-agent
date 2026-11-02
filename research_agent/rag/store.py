"""Data helper: the Chroma vector store that holds the annual-report chunks."""

import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from research_agent.config import COLLECTION_NAME, embedding_model

load_dotenv()

logger = logging.getLogger(__name__)

CHROMA_DIR = Path(__file__).resolve().parents[2] / "data" / "chroma"
# Chunks per add_documents call. 100 chunks of up to 3,200 chars is about 80,000 tokens,
# well under OpenAI's 300,000-token limit for one embeddings request.
BATCH_SIZE = 100


def get_store(collection_name: str = COLLECTION_NAME) -> Chroma:
    """Return the named Chroma collection saved in data/chroma, embedding with EMBEDDING_MODEL from .env."""
    embeddings = OpenAIEmbeddings(model=embedding_model())
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def chunk_id(chunk: Document) -> str:
    """Fixed ID from metadata, e.g. 'tcs-fy2025-26-p181-c2', so re-adding a chunk replaces it."""
    m = chunk.metadata
    return f"{m['company']}-{m['fy']}-p{m['pdf_page']}-c{m['chunk']}".lower()


def add_chunks(
    chunks: list[Document], collection_name: str = COLLECTION_NAME
) -> Chroma:
    """Embed and save chunks in batches of BATCH_SIZE. Chroma upserts by ID, so nothing is duplicated."""
    store = get_store(collection_name)
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        store.add_documents(batch, ids=[chunk_id(c) for c in batch])
        logger.info(
            "Added chunks %d-%d of %d", start + 1, start + len(batch), len(chunks)
        )
    return store
