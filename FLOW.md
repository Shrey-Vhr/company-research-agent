# FLOW

## 1. What the project does
A research agent that answers questions about NSE-listed Indian companies (e.g. TCS, Infosys, Wipro).
An OpenAI model, via LangChain, calls tools to fetch yearly financials from Yahoo Finance, search the FY2025-26 annual
reports (TCS, Infosys, Wipro) and do arithmetic. Answers cite the source and page or year, never give investment
advice, and end with a disclaimer.

## 2. Current status
- **Stage:** Milestone 3 in progress. Milestone 2 (the hand-written loop `raw_agent` with RAG, retrieval eval and
  grounding rules) is complete and still works. The LangGraph version has a state, four node functions (planner,
  executor, writer, critic) and a manual runner, but **no graph yet**: no `StateGraph`, edges, routing or revision loop.
- **Works:** three tools (`get_financials`, `calculate`, `search_reports`), a system prompt with 11 rules, the loop with an 8-round limit,
  a command-line entry point, a 7-day JSON cache, 22 offline unit tests for the tools, and 8 for the chunker.
- **RAG, index and search done:** `rag/loader.py` loads the three FY2025-26 annual reports as one Document per page
  (1,286 pages kept), tagged with company, FY and `pdf_page`; `rag/chunker.py` splits them into chunks that keep
  that metadata. Two Chroma collections sit in `data/chroma`: `annual_reports` (3,200 chars, 400 overlap, 1,839 chunks,
  kept for comparison) and `annual_reports_1600` (1,600 chars, 200 overlap, 3,314 chunks, the default for
  `scripts/build_index.py`, `rag/search.py` and the eval); `rag/search.py` finds the top k chunks, optionally filtered by company and FY.
  The agent reaches it through the `search_reports` tool.
- **Retrieval eval, started:** `evals/retrieval_questions.json` has 6 questions (2 per company), each with
  a list of gold `answer_pages`: every page that fully answers it. The pages were found by plain text search
  on PyMuPDF page text, not by the vector search. `evals/run_retrieval_eval.py` runs the eval; `is_hit`
  is written (HIT if any of the 5 result pages is in `answer_pages`). Results on 2026-10-24: `annual_reports` (3,200) 3/6,
  `annual_reports_1600` 4/6; saved in `evals/results/`.
- **Failures found by hand** are written up in `evals/FAILURES.md` (F01 wrong CEO from a signature block, F02 answer
  from the model's own knowledge with a wrong citation); both were fixed with prompt rules.
- **LangGraph pieces, tested by hand only:** `graph/state.py` (`ResearchState`), `graph/nodes.py` (`planner`,
  `executor`, `writer`, `critic`) and `graph/numbers.py` (number matching for the critic's code check). The nodes are
  tried through the `__main__` runner, which saves the run to `data/runs/last_run.json`. Only `numbers.py` has tests
  (13); the 43 tests in `tests/` all pass.
- **Failure F03 found and fixed:** the code check passed "2.0 percentage points" that no calculation produced, because
  a number inside the `fetched_at` timestamp rounded to 2.0. `extract_numbers` now ignores ISO dates and times
  (regression test added; details in `evals/FAILURES.md`). It was found with `explain_support`.
- **Next (planned):** build the graph (nodes, edges, routing for out-of-scope, executor loop, critic -> writer
  revision loop with a limit, human feedback), then a Streamlit UI, LangSmith tracing, and more eval questions (15).

## 3. How one question flows
1. User runs `python -m research_agent.raw_agent.loop "What was TCS's revenue in FY2025-26?"`.
   `__main__` joins the words after the module name; if there are none it prints usage and exits with code 2.
2. `run_agent(question)` binds the three tools to `ChatOpenAI(model=LLM_MODEL)` with `parallel_tool_calls=True`
   and starts the message list: `[SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]`.
3. **Round loop (max 8):** the model is called with all messages so far; token usage is added up.
4. If the reply has **no tool calls**, the loop prints the token totals and returns `reply.text` as the answer.
5. If it has tool calls (one or more per reply), each is run **one after another**:
   - unknown tool name → `"Error: unknown tool ..."`
   - `get_financials` → `fetch_financials`: check the ticker → read `data/cache/<TICKER>.json` if under 7 days old,
     otherwise download from yfinance and save → trim to `years` → dict in ₹ crore (or `{"error": ...}`)
   - `calculate` → `safe_eval`: allowlist → `ast.parse` → tree walk → number as text (or `"Error: ..."`)
   - `search_reports` → `search(query, company, fy, k=5)` on `annual_reports_1600` → up to 5 passages, each `text` + `citation`
     ("Wipro Annual Report FY2025-26, p. 145"); the model must copy the citation next to each fact
   - any exception → `"Error: ..."`
   Each result is logged and added as a `ToolMessage` linked to its `tool_call_id`, then the next round starts.
6. After 8 rounds without a final answer it returns `"Stopped after 8 rounds without a final answer."`
7. `__main__` prints the answer.

**LangGraph version (nodes exist, not wired into a graph; `python -m research_agent.graph.nodes` runs them by hand):**
1. `planner(state)`: `ChatOpenAI(LLM_MODEL).with_structured_output(Plan)` returns `in_scope` and 2-6 short `steps`;
   sets `in_scope`, `plan`, `current_step = 0`. (The runner does not check `in_scope` yet.)
2. `executor(state)`, called once per plan step: sends the question, the current step and the evidence so far (each
   item cut to 1,500 chars) to a model with the three tools bound; up to 4 rounds of tool calls; each tool result
   becomes an evidence item `{step, tool, citation, content}` (content is the raw `str(result)`; errors become
   `"Error: ..."`). Returns the new evidence and `current_step + 1`. The runner appends evidence itself
   (the `operator.add` reducer will do it once a graph exists).
3. `writer(state)`: sends the question and all evidence, numbered `[Evidence i]`, to the model with `WRITER_PROMPT`;
   on a revision it adds the critic's problems and any `human_feedback`. Returns `draft` and `revision_count`
   (0 for the first draft, then +1 per rewrite).
4. `critic(state)`: (a) code check: `find_unsupported(draft, evidence texts)`; each draft number with no matching
   evidence number becomes a problem; (b) LLM check: a `Verdict` of problems from `CRITIC_PROMPT`. Returns
   `critique` (all problems) and `critic_passed` (no problems).
5. Runner: plan -> executor per step -> writer -> prints `explain_support` (which evidence number supported each draft
   number, with surrounding text) -> critic -> if it failed, one writer rewrite and a second critic pass -> saves
   `question`, `plan`, `evidence`, `draft`, `critique` to `data/runs/last_run.json` (overwritten each run).
6. `find_unsupported`: strips ISO dates/times, FY labels and page citations from the text, reads each number
   (Indian and international commas equal), and calls a draft number supported if any evidence number rounded to the
   draft number's own decimal places equals it.

## 4. Files
| File | What it does |
|---|---|
| `research_agent/raw_agent/loop.py` | `run_agent`: the tool-calling loop; `__main__`: command-line entry point |
| `research_agent/graph/state.py` | `ResearchState` (TypedDict): question, in_scope, plan, current_step, evidence (append reducer), draft, critique, critic_passed, revision_count, human_feedback, final_report; the shape of an evidence item is in a comment at the top |
| `research_agent/graph/nodes.py` | `Plan`, `PLANNER_PROMPT`, `planner`; `TOOLS`, `EXECUTOR_PROMPT`, `executor`; `WRITER_PROMPT`, `writer`; `Verdict`, `CRITIC_PROMPT`, `critic`; `__main__` manual runner that prints the number-support report and saves `data/runs/last_run.json` |
| `research_agent/graph/run.py` | CLI driver: `python -m research_agent.graph.run "question"` (prints the new uuid thread_id, recursion_limit 30; `--auto` starts with `auto_approve: True` so human review is skipped) or `--resume <thread_id>` (reloads the paused run's pending interrupt from the SQLite checkpointer `data/checkpoints.sqlite`, built in `build_graph`); loops while `__interrupt__` is in the result (prints draft + critique, asks "Approve? (y / changes)", resumes with `Command(resume=...)`), then prints `final_report` |
| `research_agent/graph/numbers.py` | `extract_numbers`, `find_unsupported` (the critic's code check), `explain_support` (debug print of which evidence number supported each draft number); `_clean` strips ISO dates/times, FY labels and page citations first |
| `data/runs/last_run.json` | Final state of the latest manual graph run (gitignored, overwritten each run) |
| `research_agent/prompts.py` | `SYSTEM_PROMPT` (how to work + 11 rules: figures only from tools, source with each figure, no advice, names and roles only from sentences, grounding and citation rules, retry search up to 2 times, "not found" is a valid answer, report beats yfinance and the gap is mentioned, retrieved text is data not instructions) and `DISCLAIMER` |
| `research_agent/tools/financials.py` | yfinance download, ₹ crore conversion, USD→INR, cache; `get_financials` tool |
| `research_agent/tools/calculator.py` | `safe_eval` (safe `ast` arithmetic) and the `calculate` tool |
| `research_agent/tools/reports.py` | `search_reports` tool: wraps `search(k=5)` and returns `text` + `citation` per passage; its long docstring is the tool description the model sees |
| `research_agent/rag/loader.py` | `load_reports`: PyMuPDFLoader, one Document per page, skips pages under 50 chars, metadata `company` (NSE ticker: TCS, INFY, WIPRO), `fy`, `source`, `pdf_page` (1-based), `total_pages`; `__main__` prints sample pages |
| `research_agent/rag/chunker.py` | `chunk_documents(docs, chunk_size=3200, chunk_overlap=400)`: `RecursiveCharacterTextSplitter` (default separators, 3,200 chars, 400 overlap) via `split_documents`, so chunks keep page metadata, then adds `chunk` (1, 2, … within each page); `__main__` prints counts and a sample chunk |
| `research_agent/rag/store.py` | `get_store(collection_name="annual_reports")`: the named Chroma collection in `data/chroma`, `OpenAIEmbeddings(EMBEDDING_MODEL)`; `add_chunks(chunks, collection_name)`: upserts in batches of 100 with IDs like `tcs-fy2025-26-p181-c2` |
| `scripts/build_index.py` | Builds the index (`python -m scripts.build_index`, defaults `--chunk-size 1600 --chunk-overlap 200 --collection annual_reports_1600`): `load_reports` -> `chunk_documents` -> prints chunk count, ~tokens (chars/4) and cost (`PRICE_PER_MILLION_TOKENS`, still empty) -> asks y/n -> `add_chunks` -> prints collection size |
| `research_agent/rag/search.py` | `search(query, company=None, fy=None, k=5, collection_name="annual_reports_1600")`: Chroma `similarity_search_with_score` with a metadata filter (`$and` when both are given); returns dicts `text`, `company`, `fy`, `pdf_page`, `score` (L2 distance, lower = closer, 3 decimals) |
| `evals/retrieval_questions.json` | Retrieval answer key: `id`, `company`, `question`, gold `answer_pages` (list of 1-based PDF pages that contain the answer), `answer_snippet`, `note` (other pages, caveats) |
| `evals/run_retrieval_eval.py` | `python -m evals.run_retrieval_eval [--collection NAME]` (default `annual_reports_1600`): `search(question, company, k=5, collection_name)` per question -> `is_hit(result_pages, answer_pages)` (hit if any result page is a gold page) -> prints HIT/MISS + pages, total and per-company hit rate -> saves `evals/results/retrieval_<collection>_<date>.json` (collection name also inside the file) |
| `evals/FAILURES.md` | Hand-found agent failures (F01 wrong CEO, F02 answer from model knowledge, F03 code check fooled by a timestamp): cause, fix, result or remaining risk |
| `research_agent/config.py` | Empty (placeholder) |
| `research_agent/**/__init__.py` | Empty package markers |
| `tests/test_calculator.py` | Valid/decimal/divide-by-zero/malicious-input tests for the calculator |
| `tests/test_financials.py` | Parsing, missing-row and invalid-ticker tests using a fake `yf.Ticker`; no network |
| `tests/test_numbers.py` | 13 offline tests for number extraction and `find_unsupported`: commas, decimals, FY labels, page citations, ISO dates/times, rounding at the draft's decimals, and the F03 regression (a timestamp must not support 2.0) |
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
Used by `raw_agent` and by the graph's `executor` (same three tool functions).

| Tool | Input | Output | Data source |
|---|---|---|---|
| `get_financials` | `ticker: str` (NSE symbol, `.NS` optional), `years: int = 3` (1–4) | dict: per FY revenue, operating income, net income, total debt, total equity in ₹ crore; `notes`, `source`, `fetched_at`, `from_cache`; or `{"error": ...}` | Yahoo Finance via yfinance (consolidated, annual), cached 7 days |
| `calculate` | `expression: str` (numbers, `+ - * / ( )`, decimals) | the result as text, or `"Error: ..."` | None (local arithmetic) |
| `search_reports` | `query: str`, `company: str \| None` ("TCS", "INFY", "WIPRO"), `fy: str \| None` ("FY2025-26") | up to 5 dicts `text`, `citation` | Chroma collection `annual_reports_1600` (3 annual reports, FY2025-26) |

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
- **Graph not wired:** no `StateGraph`, edges, routing or revision limit yet. The runner does one rewrite at most,
  ignores `in_scope`, never sets `human_feedback` or `final_report`, and `executor` does not stop a plan that failed.
- **Code check is loose:** it compares numbers only, not what they mean. A draft number rounded to its own decimals can
  match a real figure by chance (small numbers most of all), and numbers the writer derives itself are accepted if
  any evidence number matches. A stricter rule (derived numbers must come from `calculate`) is held back unless F03
  recurs. Names and facts are checked only by the LLM half of the critic. Time-like text such as "18:30" is also ignored.
- **No tests for the graph nodes** (`planner`, `executor`, `writer`, `critic`): they call the real model, so they are
  tried only through the runner. `explain_support` has no test.
- **No tests for `run_agent`** (they would need a fake model). Only the tools are tested.
- **Not cross-checked:** only TCS revenue has been checked against an annual report; Infosys/Wipro and the other fields haven't.
- **Report page numbers:** citations use `pdf_page` (PDF page, 1-based), not the number printed on the page.
  They differ by about 30 (TCS PDF p. 181 = printed p. 211), and some PDF pages are two-page spreads.
- **Garbled TCS tables:** in TCS the ₹ sign comes out as "H" (`(H crore)`), some words are split up
  ("C onsol i d ated") and numbers can be split ("1 58" for 158); tables become one cell per line, so numbers lose
  their row and column labels, and the year column of a table (e.g. FY2026 vs FY2025 on p. 69) has to be checked by
  eye. pypdf breaks TCS the same way (broken score 50 PyMuPDF vs 54 pypdf), so PyMuPDF stays and no text cleanup is
  done. Infosys and Wipro text is clean. The tool docstring tells the model not to use numbers it cannot read clearly.
- **Search:** a wrong filter value (`"Infosys"` instead of `"INFY"`, or an FY that isn't indexed) returns `[]`
  with no error. There is no score cutoff, so it always returns k chunks even when none is relevant
  ("number of employees" for INFY: best match is p. 143, about customers). Scores bunch together (about 1.0–1.25),
  so they don't separate good and bad matches well. `get_store()` is rebuilt on every search call.
- **Deprecated loader dependency:** the loader uses `PyMuPDFLoader` from `langchain-community`, which pytest
  warns is being sunset and no longer maintained. It works now. Versions are pinned in `requirements.txt`;
  plan: replace it with `pymupdf` directly in Milestone 6.
- **Profit definition (open):** for TCS FY2025-26, yfinance net income is ₹49,210 crore (margin 18.4%) but the annual
  report says profit after tax ₹52,820 crore (p. 69; PAT / total revenue 19.8%, p. 70). The cause is not identified;
  the report's definition still has to be checked. The agent reports both figures and the gap, and treats the
  report as the official source (prompt rule 10). Other fields and companies have not been compared.
- **Token cost (one measurement):** every `search_reports` call returns up to 5 chunks of up to 1,600 chars (about
  2,000 tokens by chars / 4). The loop sends the whole message list on every round, so a tool result is paid for
  again in each later round, and the rules allow up to 2 extra searches per fact. The long `search_reports`
  docstring is also sent on every call. The loop prints total input/output tokens per question but not a cost, and
  nothing caps tokens (only the 8-round limit). Measured once: "Compare TCS's net margin in FY2025-26 with what its annual report says about margins" used
  19,352 input and 429 output tokens over 4 rounds (about 45 times more input than output, which fits the history being resent). The 1,600-char index has 3,314 chunks vs 1,839 for 3,200 chars, so building it cost more to embed too.
