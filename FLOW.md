# FLOW

## 1. What the project does
A research agent that answers questions about NSE-listed Indian companies (e.g. TCS, Infosys, Wipro).
An OpenAI model, via LangChain, calls tools to fetch yearly financials from Yahoo Finance and do arithmetic.
Answers cite the source and financial year, never give investment advice, and end with a disclaimer.

## 2. Current status
- **Stage:** hand-written tool-calling loop (`raw_agent`), before LangGraph, RAG, UI and evals.
- **Works:** two tools (`get_financials`, `calculate`), a system prompt, the loop with an 8-round limit,
  a command-line entry point, a 7-day JSON cache, 22 offline unit tests for the tools, and 8 for the chunker.
- **RAG, index and search done:** `rag/loader.py` loads the three FY2025-26 annual reports as one Document per page
  (1,286 pages kept), tagged with company, FY and `pdf_page`; `rag/chunker.py` splits them into chunks that keep
  that metadata. Two Chroma collections sit in `data/chroma`: `annual_reports` (3,200 chars, 400 overlap, 1,839 chunks,
  kept for comparison) and `annual_reports_1600` (1,600 chars, 200 overlap, 3,314 chunks, the default for
  `scripts/build_index.py`, `rag/search.py` and the eval); `rag/search.py` finds the top k chunks, optionally filtered by company and FY.
  RAG is not connected to the agent yet (no retrieval tool).
- **Retrieval eval, started:** `evals/retrieval_questions.json` has 6 questions (2 per company), each with
  a list of gold `answer_pages`: every page that fully answers it. The pages were found by plain text search
  on PyMuPDF page text, not by the vector search. `evals/run_retrieval_eval.py` runs the eval; `is_hit`
  is written (HIT if any of the 5 result pages is in `answer_pages`). Results on 2026-10-24: `annual_reports` (3,200) 3/6,
  `annual_reports_1600` 4/6; saved in `evals/results/`.
- **Next (planned, from requirements.txt):** a retrieval tool for the agent, a LangGraph version
  of the agent, a Streamlit UI, LangSmith tracing and evals.

## 3. How one question flows
1. User runs `python -m research_agent.raw_agent.loop "What was TCS's revenue in FY2025-26?"`.
   `__main__` joins the words after the module name; if there are none it prints usage and exits with code 2.
2. `run_agent(question)` binds both tools to `ChatOpenAI(model=LLM_MODEL)` with `parallel_tool_calls=True`
   and starts the message list: `[SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]`.
3. **Round loop (max 8):** the model is called with all messages so far; token usage is added up.
4. If the reply has **no tool calls**, the loop prints the token totals and returns `reply.text` as the answer.
5. If it has tool calls (one or more per reply), each is run **one after another**:
   - unknown tool name → `"Error: unknown tool ..."`
   - `get_financials` → `fetch_financials`: check the ticker → read `data/cache/<TICKER>.json` if under 7 days old,
     otherwise download from yfinance and save → trim to `years` → dict in ₹ crore (or `{"error": ...}`)
   - `calculate` → `safe_eval`: allowlist → `ast.parse` → tree walk → number as text (or `"Error: ..."`)
   - any exception → `"Error: ..."`
   Each result is logged and added as a `ToolMessage` linked to its `tool_call_id`, then the next round starts.
6. After 8 rounds without a final answer it returns `"Stopped after 8 rounds without a final answer."`
7. `__main__` prints the answer.

## 4. Files
| File | What it does |
|---|---|
| `research_agent/raw_agent/loop.py` | `run_agent`: the tool-calling loop; `__main__`: command-line entry point |
| `research_agent/prompts.py` | `SYSTEM_PROMPT` (four rules + answer format) and `DISCLAIMER` |
| `research_agent/tools/financials.py` | yfinance download, ₹ crore conversion, USD→INR, cache; `get_financials` tool |
| `research_agent/tools/calculator.py` | `safe_eval` (safe `ast` arithmetic) and the `calculate` tool |
| `research_agent/rag/loader.py` | `load_reports`: PyMuPDFLoader, one Document per page, skips pages under 50 chars, metadata `company` (NSE ticker: TCS, INFY, WIPRO), `fy`, `source`, `pdf_page` (1-based), `total_pages`; `__main__` prints sample pages |
| `research_agent/rag/chunker.py` | `chunk_documents(docs, chunk_size=3200, chunk_overlap=400)`: `RecursiveCharacterTextSplitter` (default separators, 3,200 chars, 400 overlap) via `split_documents`, so chunks keep page metadata, then adds `chunk` (1, 2, … within each page); `__main__` prints counts and a sample chunk |
| `research_agent/rag/store.py` | `get_store(collection_name="annual_reports")`: the named Chroma collection in `data/chroma`, `OpenAIEmbeddings(EMBEDDING_MODEL)`; `add_chunks(chunks, collection_name)`: upserts in batches of 100 with IDs like `tcs-fy2025-26-p181-c2` |
| `scripts/build_index.py` | Builds the index (`python -m scripts.build_index`, defaults `--chunk-size 1600 --chunk-overlap 200 --collection annual_reports_1600`): `load_reports` -> `chunk_documents` -> prints chunk count, ~tokens (chars/4) and cost (`PRICE_PER_MILLION_TOKENS`, still empty) -> asks y/n -> `add_chunks` -> prints collection size |
| `research_agent/rag/search.py` | `search(query, company=None, fy=None, k=5, collection_name="annual_reports_1600")`: Chroma `similarity_search_with_score` with a metadata filter (`$and` when both are given); returns dicts `text`, `company`, `fy`, `pdf_page`, `score` (L2 distance, lower = closer, 3 decimals) |
| `evals/retrieval_questions.json` | Retrieval answer key: `id`, `company`, `question`, gold `answer_pages` (list of 1-based PDF pages that contain the answer), `answer_snippet`, `note` (other pages, caveats) |
| `evals/run_retrieval_eval.py` | `python -m evals.run_retrieval_eval [--collection NAME]` (default `annual_reports_1600`): `search(question, company, k=5, collection_name)` per question -> `is_hit(result_pages, answer_pages)` (hit if any result page is a gold page) -> prints HIT/MISS + pages, total and per-company hit rate -> saves `evals/results/retrieval_<collection>_<date>.json` (collection name also inside the file) |
| `research_agent/config.py` | Empty (placeholder) |
| `research_agent/**/__init__.py` | Empty package markers |
| `tests/test_calculator.py` | Valid/decimal/divide-by-zero/malicious-input tests for the calculator |
| `tests/test_financials.py` | Parsing, missing-row and invalid-ticker tests using a fake `yf.Ticker`; no network |
| `tests/test_chunker.py` | 8 offline tests for `chunk_documents` on small fake Documents (size limit, overlap, metadata, no page mixing, chunk numbering per page and company, short page, no empty chunks); no PDFs, API or Chroma |
| `tests/fixtures/tcs_sample.json` | Two years of TCS statements in raw rupees, shaped like yfinance |
| `playground/*.py` | Learning experiments (yfinance rows, a first tool-calling script, `compare_loaders.py` PyMuPDF vs pypdf); not used by the app |
| `data/reports/` | The three annual-report PDFs (gitignored; sources in `docs/DATA_SOURCES.md`) |
| `docs/DATA_SOURCES.md` | Where each report PDF came from, download date, page count |
| `data/cache/` | Cached `get_financials` results, one JSON per ticker (gitignored) |
| `.env` | `OPENAI_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, LangSmith settings (gitignored) |
| `requirements.txt` | Dependencies pinned to exact versions (`==`, top-level packages only), including ones for later stages (LangGraph, Streamlit) |
| `CLAUDE.md` / `DECISIONS.md` / `FLOW.md` | Working rules / decision log / this file |
| `README.md` | Title only |

## 5. Tools
| Tool | Input | Output | Data source |
|---|---|---|---|
| `get_financials` | `ticker: str` (NSE symbol, `.NS` optional), `years: int = 3` (1–4) | dict: per FY revenue, operating income, net income, total debt, total equity in ₹ crore; `notes`, `source`, `fetched_at`, `from_cache`; or `{"error": ...}` | Yahoo Finance via yfinance (consolidated, annual), cached 7 days |
| `calculate` | `expression: str` (numbers, `+ - * / ( )`, decimals) | the result as text, or `"Error: ..."` | None (local arithmetic) |

## 6. Known limits and open problems
- **Tool timing is always logged as 0.00s:** `start` is recorded but `seconds` is never computed from it.
- **Tools run one at a time:** `parallel_tool_calls=True` lets the model *request* several calls in one reply,
  but the loop runs them one by one.
- **The model sees Python output, not JSON:** `ToolMessage(content=str(result))` turns `get_financials`'s dict
  into Python text, so missing values appear as `None`, while the docstring says "null".
- **Token totals aren't printed when the 8-round limit is hit.** `ChatOpenAI` is rebuilt on every call, and a
  missing `LLM_MODEL` raises a `KeyError`.
- **Data:** only about 4 years of annual figures, with no quarterly data, prices or news. Infosys is
  converted from USD at the yearly average rate, so its figures are approximate. FY labels assume a March year-end.
- **Repeatability:** the cache expires after 7 days, so eval answers can change when Yahoo updates its data.
- **Calculator:** no `**`, so it can't do CAGR.
- **Retrieval eval is small and uneven:** 6 questions; `answer_pages` ranges from 1 page (r04, r05) to 8 (r03),
  so some questions are easier to hit and per-company hit rates are not fully comparable.
  No cross-company or unanswerable questions yet. The 1,600 vs 3,200 chunk result (4/6 vs 3/6) is one question
  out of six: it points one way but 6 questions can't prove it; the plan calls for 15.
- **No tests for `run_agent`** (they would need a fake model). Only the tools are tested.
- **Not cross-checked:** only TCS revenue has been checked against an annual report; Infosys/Wipro and the other fields haven't.
- **Report page numbers:** citations use `pdf_page` (PDF page, 1-based), not the number printed on the page.
  They differ by about 30 (TCS PDF p. 181 = printed p. 211), and some PDF pages are two-page spreads.
- **PDF text defects:** in TCS the ₹ sign comes out as "H" (`(H crore)`) and some words are split up
  ("C onsol i d ated"); tables become one cell per line, so numbers lose their row and column labels.
  pypdf breaks TCS the same way (broken score 50 PyMuPDF vs 54 pypdf), so PyMuPDF stays and no cleanup is done
  until step 7 shows whether it hurts TCS retrieval. Infosys and Wipro text is clean.
- **Search:** a wrong filter value (`"Infosys"` instead of `"INFY"`, or an FY that isn't indexed) returns `[]`
  with no error. There is no score cutoff, so it always returns k chunks even when none is relevant
  ("number of employees" for INFY: best match is p. 143, about customers). Scores bunch together (about 1.0–1.25),
  so they don't separate good and bad matches well. `get_store()` is rebuilt on every search call.
- **Deprecated loader dependency:** the loader uses `PyMuPDFLoader` from `langchain-community`, which pytest
  warns is being sunset and no longer maintained. It works now. Versions are pinned in `requirements.txt`;
  plan: replace it with `pymupdf` directly in Milestone 6.
