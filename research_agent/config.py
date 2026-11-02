"""Settings in one place. The defaults are the values the project was built and evaluated with.

The model names must be set in .env (no defaults); the collection name and the daily cap can be overridden there.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Models


def _require(name: str) -> str:
    """Return the environment variable `name`, or raise an error saying what to set in .env."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add a line {name}=<model name> to your .env file."
        )
    return value


def llm_model() -> str:
    """Chat model name from LLM_MODEL in .env."""
    return _require("LLM_MODEL")


def embedding_model() -> str:
    """Embedding model name from EMBEDDING_MODEL in .env."""
    return _require("EMBEDDING_MODEL")


# Agent limits
RAW_MAX_ROUNDS = 8  # raw loop: model calls before it gives up
EXECUTOR_MAX_ROUNDS = 4  # graph executor: model calls for one plan step
MAX_REVISIONS = 2  # graph: critic-triggered rewrites before moving on
RECURSION_LIMIT = 30  # LangGraph: max node steps in one run

# Retrieval
CHUNK_SIZE = 1600  # characters per chunk when building the index
CHUNK_OVERLAP = 200
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "annual_reports_1600")

# App
MAX_RUNS_PER_DAY = int(os.getenv("MAX_RUNS_PER_DAY", "20"))
