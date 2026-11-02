# FLOW

## 1. What the project does
A research agent that answers questions about NSE-listed Indian companies (e.g. TCS, Infosys, Wipro).
An OpenAI model, via LangChain and a LangGraph workflow, calls tools to fetch yearly financials from Yahoo Finance, search the FY2025-26 annual
reports (TCS, Infosys, Wipro) and do arithmetic. Answers cite the source and page or year, never give investment
advice, and end with a disclaimer. A Streamlit app (`app.py`) lets a person ask a question, watch the plan and
evidence, approve or change the draft, and download the final report.

## 2. Current status
- **Stage:** Milestones 1 to 4 complete. **Milestone 5, evals** (raw agent vs graph without critic vs full graph):
  answer key (`evals/questions.json`), runner (`evals/run_evals.py`, with `score_numbers`) and summary script (`evals/summarize.py`) are done.
  One full run (20 questions x 3 versions, 1 repeat, 2026-10-30_192104) is scored and its 27 text answers (q12-q20) are
  hand-graded (drafted by Claude Code, all 27 agreed). Strict / percentage-tolerant score, out of 20: raw 19 / 19,
  graph_no_critic 18 / 20, graph_full 20 / 20; no run errored. Average per question: raw 9.1 s and $0.0006, graph_no_critic
  21.2 s and $0.0015, graph_full 28.6 s and $0.0025 (prices set in `summarize.py`). The only strict failures are q19 raw
  (gave an HCLTech revenue, F06) and q09/q10 graph_no_critic (percent format, F05). A second run with 3 repeats
  (`run_2026-11-01_195047.json`, 33 of 180 results saved so far) is not summarized yet. See `DECISIONS.md` 2026-11-01.
  Milestone 2 (the hand-written loop `raw_agent` with RAG, retrieval eval and grounding rules) still works. The
  Milestone 3 graph works end to end: planner, executor, writer, critic, human review with interrupt and resume, a
  SQLite checkpointer, the `auto_approve` flag and LangSmith tracing. Milestone 4 added the Streamlit app (`app.py`):
  run, approve, request changes, download, recovery of a paused run after a refresh or restart, a daily cap of 20 runs,
  friendly errors and input checks.
- **Works:** three tools (`get_financials`, `calculate`, `search_reports`), the raw agent (`SYSTEM_PROMPT` with 11 rules, 8-round limit),
  three entry points (raw agent and graph on the command line, and the Streamlit app), a 7-day JSON cache, 23 offline unit tests for the tools (14 calculator, 9 financials), 8 for the chunker, 16 for number extraction and 17 for routing (64 in all).
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
  from the model's own knowledge with a wrong citation, F03 timestamp fooled the code check, F04 F01 came back after
  the LangGraph rebuild); F01 and F02 were fixed with prompt rules, F03 with a code fix and test.
- **LangGraph graph:** `graph/build.py` wires `graph/nodes.py` (planner, executor, writer, critic, refuse, human_review,
  replanner, finalize) over `graph/state.py` with conditional routing; `graph/numbers.py` is the critic's number check.
  `numbers.py` has 16 tests and the routing functions have 17; the 64 tests in `tests/` all pass. The nodes call the real model and are checked by
  running the graph and reading LangSmith traces. The `__main__` runner in `nodes.py` is the old by-hand version.
- **Critic loop fix:** the critic prompt now accepts drafts that honestly state the evidence's limits (cut a Wipro CEO
  run from 64 s / 54.6K tokens to 30.5 s / 27.4K); see `DECISIONS.md` 2026-10-27.
- **Failure F03 found and fixed:** the code check passed "2.0 percentage points" that no calculation produced, because
  a number inside the `fetched_at` timestamp rounded to 2.0. `extract_numbers` now ignores ISO dates and times
  (regression test added; details in `evals/FAILURES.md`). It was found with `explain_support`.
- **Still to do:** finish and summarize the 3-repeat run (point `STAMP` in `summarize.py` at it), write up the Milestone 5
  comparison, fix the `get_financials` scope/currency problem (F06), and add more retrieval eval questions (15).

## 3. How one question flows
Diagram: `docs/graph.mmd`. Run it with `python -m research_agent.graph.run "question"` (`--auto` skips human review).
1. **Start:** `run.py` builds the graph with the SQLite checkpointer, makes a uuid `thread_id` (printed), and invokes
   it with `{"question", "evidence": []}` (plus `auto_approve: True` with `--auto`).
2. **planner:** one structured-output model call returns `in_scope` and 2-6 steps. Out of scope (advice, predictions,
   other companies) -> `refuse` writes a fixed refusal as `final_report` and the run ends.
3. **executor:** runs one plan step per visit, with the three tools bound. It sees the question, the current step and the
   evidence so far (each item cut to 1,500 chars). Each tool result is appended to `evidence` as
   `{step, tool, citation, content}`, where `content` is the raw `str(result)`. It routes back to itself until every
   step is done, then goes to the writer.
4. **writer:** drafts the report from the numbered evidence only (`WRITER_PROMPT`); on a revision it also gets the
   critic's problems and any `human_feedback`. Sets `draft` and `revision_count` (0 first, +1 per rewrite).
5. **critic:** (a) code check: every number in the draft must appear in the evidence (`find_unsupported`, rules in
   `numbers.py`); (b) LLM check of names, facts, citations, advice and disclaimer. It sets `critique` and
   `critic_passed`. A failed draft goes back to the writer while `revision_count` < 2.
6. **human_review:** reached when the critic passes, or when the 2 revisions are used up. With `auto_approve` the graph
   skips it and goes to `finalize`. Otherwise `interrupt()` pauses: `run.py` prints the draft (and the critique if
   the critic failed) and asks. "y" approves; any other text is `human_feedback` and sends the run to the replanner.
7. **replanner:** one structured-output model call (`REPLANNER_PROMPT`) sees the question, the steps already done and
   the change request, and returns only the extra steps it needs (same `Plan` shape and scope rules as the planner).
   They are appended to `plan` and `current_step` is set to the old plan length. `after_replanner`: new steps -> back
   to the executor (then writer, critic, review); empty list (wording-only or out-of-scope request) -> straight to the
   writer, which sees `human_feedback`. This costs one extra model call on every change request.
8. **finalize:** `final_report` = the draft, with a note listing the critic's remaining problems if it never passed.
   `run.py` prints it.

**Loop guards:** 4 model rounds per executor step; 2 revisions (`MAX_REVISIONS`); `recursion_limit` 30
(`RECURSION_LIMIT`; both in `research_agent/config.py`). **Persistence:** every step is saved under the `thread_id` in `data/checkpoints.sqlite`; a run
paused at human review can be continued later with `--resume <thread_id>`. **Baseline:** the older hand-written loop in
`research_agent/raw_agent/` (one model, three tools, 8 rounds, `SYSTEM_PROMPT`) is kept for comparison and still works:
`python -m research_agent.raw_agent.loop "question"`. The tool functions are shared by both.

**Through the Streamlit app** (`streamlit run app.py`; the script re-runs from the top on every click):
1. **Page load:** `build_graph()` is cached with `st.cache_resource`. `st.session_state.thread_id` is taken from the
   URL (`?thread=...`) if present, so a refresh or a Streamlit restart finds the same run. Nothing else is kept in
   `session_state` except the question and feedback text boxes.
2. **Run:** rejects an empty question, one over 500 characters, or a day with 20 runs already counted in
   `data/usage.sqlite`. Otherwise it counts the run, makes a new uuid `thread_id`, writes it to the URL, and calls
   `graph.invoke({"question", "evidence": []})`. The call returns when the graph reaches the `human_review` pause (or
   finishes, for a refusal).
3. **Every rerun:** the page calls `graph.get_state(thread config)` and draws the plan, the evidence table (step, tool,
   source), the draft with the critic's problems while the run is paused (`snapshot.next` is not empty), and the
   final report with a download button once `final_report` exists.
4. **Approve / Send:** resume with `graph.invoke(Command(resume={"action": "approve"}))` or
   `Command(resume={"action": "revise", "feedback": ...})`, then `st.rerun()` to redraw from the checkpointer.
5. **Errors:** every `graph.invoke` goes through `safe_invoke`; on an exception the page shows a short message and the
   exception type only, and the full traceback is printed to the terminal. Model output is shown with `st.markdown`
   without `unsafe_allow_html`, so HTML in an answer is displayed as text.

**Through the evals** (`python -m evals.run_evals`, then `python -m evals.summarize`):
1. **Run:** for each question and version, `run_one` calls the raw loop (`run_agent`) or `graph.invoke` (new `thread_id`,
   `auto_approve: True`, `skip_critic` True for `graph_no_critic`). An exception is recorded as `error` and the run goes on.
   Everything is saved to `evals/results/run_<stamp>.json` after every run, plus a `grading_<stamp>.csv` for q12-q20.
2. **Grade:** q12-q20 are graded by hand in the CSV's `pass` column (1/0) using the pass rules in `questions.json`.
3. **Summarize:** q01-q11 are scored by `score_numbers` (strict) and by `score_numbers_percent_ok` (also accepts "22.9%"
   as 0.229, plain-ratio questions only); q12-q20 use the hand grade in both. `summarize.py` writes the per-version table
   and the failed-questions table to `summary_<stamp>.md`.

## 4. Files
| File | What it does |
|---|---|
| `research_agent/raw_agent/loop.py` | `run_agent(question, return_details=False)`: the tool-calling loop (with `return_details=True` it returns `(text, {tool_results, input_tokens, output_tokens})`, used by the evals); `__main__`: command-line entry point. Round and timing logs go through `logging` at INFO (per-tool-call seconds measured with `time.perf_counter()`); `__main__` and `run_evals.py` turn logging on |
| `research_agent/graph/state.py` | `ResearchState` (TypedDict): question, in_scope, plan, current_step, evidence (append reducer), draft, critique, critic_passed, revision_count, human_feedback, final_report, auto_approve; the shape of an evidence item is in a comment at the top |
| `research_agent/graph/build.py` | `build_graph(checkpointer=None)`: adds the 8 nodes and edges, routing functions `after_planner`, `after_executor`, `after_critic`, `after_human`, `after_replanner`, `MAX_REVISIONS = 2`; `default_checkpointer()` opens the SQLite checkpointer |
| `research_agent/graph/nodes.py` | `Plan`, `PLANNER_PROMPT`, `planner`; `TOOLS`, `EXECUTOR_PROMPT`, `executor`; `WRITER_PROMPT`, `writer`; `Verdict`, `CRITIC_PROMPT`, `critic`; `refuse`, `human_review` (`interrupt`), `REPLANNER_PROMPT`, `replanner`, `finalize`; `__main__` manual runner (no graph) that prints the number-support report and saves `data/runs/last_run.json` |
| `research_agent/graph/run.py` | CLI driver: `python -m research_agent.graph.run "question"` (prints the new uuid thread_id, recursion_limit 30; `--auto` starts with `auto_approve: True` so human review is skipped; `--no-critic` starts with `skip_critic: True`, for the eval comparison) or `--resume <thread_id>` (reloads the paused run's pending interrupt from the SQLite checkpointer `data/checkpoints.sqlite`, built in `build_graph`); loops while `__interrupt__` is in the result (prints draft + critique, asks "Approve? (y / changes)", resumes with `Command(resume=...)`), then prints `final_report` |
| `research_agent/graph/numbers.py` | `extract_numbers`, `find_unsupported` (the critic's code check), `explain_support` (debug print of which evidence number supported each draft number); `_clean` strips ISO dates/times, FY labels and page citations first |
| `data/checkpoints.sqlite` | LangGraph checkpoints, one thread per run; used by the CLI and the app (gitignored via `data/`) |
| `data/usage.sqlite` | Table `usage(date, count)`: how many runs the app has started today, for the daily cap (gitignored via `data/`) |
| `data/runs/last_run.json` | Final state of the latest manual `nodes.py` run (gitignored, overwritten each run) |
| `research_agent/prompts.py` | `SYSTEM_PROMPT` (how to work + 11 rules: figures only from tools, source with each figure, no advice, names and roles only from sentences, grounding and citation rules, retry search up to 2 times, "not found" is a valid answer, report beats yfinance and the gap is mentioned, retrieved text is data not instructions) and `DISCLAIMER` |
| `research_agent/tools/financials.py` | yfinance download, ₹ crore conversion, USD→INR, cache; `get_financials` tool |
| `research_agent/tools/calculator.py` | `safe_eval` (safe `ast` arithmetic) and the `calculate` tool |
| `research_agent/tools/reports.py` | `search_reports` tool: wraps `search(k=5)` and returns `text` + `citation` per passage; its long docstring is the tool description the model sees |
| `research_agent/rag/loader.py` | `load_reports`: reads PDFs with `pymupdf` directly, one Document per page, skips pages under 50 chars, metadata `company` (NSE ticker: TCS, INFY, WIPRO), `fy`, `source`, `pdf_page` (1-based), `total_pages`; `__main__` prints sample pages |
| `research_agent/rag/chunker.py` | `chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)`: `RecursiveCharacterTextSplitter` (default separators, 1,600 chars, 200 overlap from `config.py`; the older 3,200/400 index was built by passing those values) via `split_documents`, so chunks keep page metadata, then adds `chunk` (1, 2, … within each page); `__main__` prints counts and a sample chunk |
| `research_agent/rag/store.py` | `get_store(collection_name=COLLECTION_NAME)`: the named Chroma collection in `data/chroma`, `OpenAIEmbeddings(EMBEDDING_MODEL)`; `add_chunks(chunks, collection_name)`: upserts in batches of 100 with IDs like `tcs-fy2025-26-p181-c2` |
| `scripts/build_index.py` | Builds the index (`python -m scripts.build_index`, defaults `--chunk-size 1600 --chunk-overlap 200 --collection annual_reports_1600`): `load_reports` -> `chunk_documents` -> prints chunk count, ~tokens (chars/4) and cost (`PRICE_PER_MILLION_TOKENS`) -> asks y/n -> `add_chunks` -> prints collection size |
| `research_agent/rag/search.py` | `search(query, company=None, fy=None, k=5, collection_name="annual_reports_1600")`: Chroma `similarity_search_with_score` with a metadata filter (`$and` when both are given); returns dicts `text`, `company`, `fy`, `pdf_page`, `score` (L2 distance, lower = closer, 3 decimals) |
| `evals/retrieval_questions.json` | Retrieval answer key: `id`, `company`, `question`, gold `answer_pages` (list of 1-based PDF pages that contain the answer), `answer_snippet`, `note` (other pages, caveats) |
| `evals/run_retrieval_eval.py` | `python -m evals.run_retrieval_eval [--collection NAME]` (default `annual_reports_1600`): `search(question, company, k=5, collection_name)` per question -> `is_hit(result_pages, answer_pages)` (hit if any result page is a gold page) -> prints HIT/MISS + pages, total and per-company hit rate -> saves `evals/results/retrieval_<collection>_<date>.json` (collection name also inside the file) |
| `evals/questions.json` | The 20 end-to-end eval questions with a hand-built answer key (Milestone 5 step 1, found without the agent): `accepted_values` (INR crore, % or ratio), `tolerance`, `expected_points`, `expected_pages` (`company` + `pdf_page`), `notes` (sources, formulas, wrong-but-tempting figures). Scored by `run_evals.score_numbers` (q01-q11) and by hand (q12-q20) |
| `evals/run_evals.py` | Eval runner and strict scorer: `python -m evals.run_evals [--versions raw graph_no_critic graph_full] [--ids q02 q18] [--repeats N]`. Runs each question on each version (graph runs: new `thread_id`, `recursion_limit` 30 (`RECURSION_LIMIT`), `auto_approve: True`, `skip_critic` True/False; tokens from `UsageMetadataCallbackHandler`; raw tokens from the loop's own totals), records answer, evidence, seconds, tokens and any error, saves `evals/results/run_<date>_<time>.json` (after every run) and `grading_<date>_<time>.csv` (q12-q20, empty `pass` column). `score_numbers(answer, item)` (strict): `extract_numbers` on the answer, then any number within `tolerance` of an `accepted_values` entry (for comparisons, every `required_groups` group needs a match) |
| `evals/summarize.py` | `python -m evals.summarize` (run with the project `.venv` Python): scores q01-q11 with `score_numbers` (strict) and `score_numbers_percent_ok` (also accepts "X%" as X/100, only for plain-ratio questions), reads the hand grades (`pass` column, q12-q20) from the grading CSV, and writes `summary_<stamp>.md`: per-version table (numbers strict and tolerant, text, overall strict and tolerant, average seconds / tokens / cost, errors) and the failed questions with a "passes % tolerant" column. `STAMP` and the token prices (`INPUT_PRICE_PER_M`, `OUTPUT_PRICE_PER_M`) are constants at the top |
| `evals/results/` | Eval outputs: `run_<stamp>.json` (all answers, evidence, seconds, tokens, errors), `grading_<stamp>.csv` (hand-graded q12-q20), `grading_<stamp>_draft.csv` (Claude Code's draft grades with `reason`, kept unchanged), `summary_<stamp>.md`, retrieval-eval JSON files, and `tests/` (earlier trial runs) |
| `evals/FAILURES.md` | Hand-found agent failures (F01 wrong CEO, F02 answer from model knowledge, F03 code check fooled by a timestamp, F04 F01 back after the rebuild, F05 ratios written as percentages, F06 raw agent answered an out-of-scope company with wrong data, F07 revise requests could not fetch new facts, F08 critic flagged a page number from a pp. list): cause, fix, result or remaining risk |
| `research_agent/config.py` | All settings: `llm_model()` / `embedding_model()` (read `LLM_MODEL` / `EMBEDDING_MODEL` from `.env`, raise `RuntimeError` naming the variable if unset), `RAW_MAX_ROUNDS` 8, `EXECUTOR_MAX_ROUNDS` 4, `MAX_REVISIONS` 2, `RECURSION_LIMIT` 30, `CHUNK_SIZE` 1600, `CHUNK_OVERLAP` 200, `COLLECTION_NAME` (`annual_reports_1600`, overridable in `.env`), `MAX_RUNS_PER_DAY` 20 (overridable in `.env`) |
| `research_agent/**/__init__.py` | Empty package markers |
| `tests/test_calculator.py` | 14 offline tests (parametrized cases counted) for the calculator: valid, decimal, divide-by-zero and malicious input |
| `tests/test_financials.py` | 9 offline tests (parametrized cases counted): parsing, missing-row and invalid-ticker tests using a fake `yf.Ticker`; no network |
| `tests/test_routing.py` | 17 offline tests for every routing function (`after_planner`, `after_executor`, `after_writer`, `after_critic`, `after_human`, `after_replanner`) on small hand-made state dicts; no model calls |
| `tests/test_numbers.py` | 16 offline tests for number extraction and `find_unsupported`: commas, decimals, FY labels, page citations (single, ranges and lists like `pp. 132–135, 166`, F08), ISO dates/times, rounding at the draft's decimals, and the F03 regression (a timestamp must not support 2.0) |
| `tests/test_chunker.py` | 8 offline tests for `chunk_documents` on small fake Documents (size limit, overlap, metadata, no page mixing, chunk numbering per page and company, short page, no empty chunks); no PDFs, API or Chroma |
| `tests/fixtures/tcs_sample.json` | Two years of TCS statements in raw rupees, shaped like yfinance |
| `playground/*.py` | Learning experiments (yfinance rows, a first tool-calling script); not used by the app |
| `data/reports/` | The three annual-report PDFs (gitignored; sources in `docs/DATA_SOURCES.md`) |
| `docs/DATA_SOURCES.md` | Where each report PDF came from, download date, page count |
| `docs/graph.mmd` | Mermaid flowchart of the graph (paste into mermaid.live) |
| `docs/trace_before.png`, `docs/trace_after.png` | LangSmith traces of the Wipro CEO run before and after the critic fix (64.4 s vs 30.5 s) |
| `data/cache/` | Cached `get_financials` results, one JSON per ticker (gitignored) |
| `app.py` | Streamlit UI (flow in section 3): question box with 3 example buttons, Run, plan / evidence / draft / final report sections, Approve and "Request changes" + Send, report download, disclaimer footer. `get_graph()` (cached), `run_config()`, `safe_invoke()`, `runs_today()` / `count_run()` (cap `MAX_RUNS_PER_DAY` from `config.py`, default 20), `?thread=` in the URL |
| `Dockerfile` | Image for the Streamlit app: `python:3.14-slim`, installs `requirements.txt` first (cached layer), copies the code, runs `streamlit run app.py` on `0.0.0.0:8501`. No keys and no `data/` inside |
| `.dockerignore` | Keeps `.env`, `.venv`, `data/`, caches, `.git`, `playground/`, `evals/results/` and `docs/*.png` out of the image |
| `.env` | `OPENAI_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, LangSmith settings (gitignored) |
| `.env.example` | Template of every variable the code reads, with empty values (copy to `.env`; no quotes around values) |
| `requirements.txt` | Dependencies pinned to exact versions (`==`, top-level packages only), including LangGraph, the SQLite checkpointer and Streamlit |
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
- **Executor does not stop a failed plan:** if a step's tool fails twice, the next steps still run and the writer
  works with whatever evidence exists.
- **Critic is now more lenient on the names-from-sentences rule;** answers may come from list rows. Retrieval still
  misses sentence passages for short exact facts (e.g. Wipro CEO on p. 19 and p. 103); hybrid search (keyword +
  vector) is the planned fix.
- **Answers vary between runs:** the same question can give different plans, evidence and wording. Evals should run
  each question more than once.
- **After editing `.env`, open a new terminal:** an open VS Code terminal keeps the old values. Close it (trash icon)
  and open a new one before running again.
- **App:** one shared daily count, no user login; the run is counted before it starts, so a failed run still counts.
  A paused run is found again only through the `?thread=` URL (or the SQLite file); there is no list of past runs.
- **Code check is loose:** it compares numbers only, not what they mean. A draft number rounded to its own decimals can
  match a real figure by chance (small numbers most of all), and numbers the writer derives itself are accepted if
  any evidence number matches. A stricter rule (derived numbers must come from `calculate`) is held back unless F03
  recurs. Names and facts are checked only by the LLM half of the critic. Time-like text such as "18:30" is also ignored.
- **No tests for the graph nodes** (`planner`, `executor`, `writer`, `critic`, `replanner`; the routing functions are tested): the nodes call the
  real model, so they are tried only by running the graph. `explain_support` has no test.
- **No tests for `run_agent`** (they would need a fake model). Only the tools are tested.
- **Cross-checked (by hand, for the eval key):** revenue and net income for all three, and debt/equity, agree with the
  reports apart from the definitions above (Infosys figures are about 0.3-0.6% off because of the USD conversion).
  Operating income has not been checked.
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
- **Profit definition (cause found, agent not changed):** for TCS FY2025-26, yfinance net income ₹49,210 crore is the
  reported profit attributable to shareholders (p. 167, p. 35). The ₹52,820 crore on p. 69 is profit *excluding
  exceptional items* (footnote on p. 69; margin 19.8%). Wipro's yfinance net income is also the equity-holders figure
  (₹13,197.4 crore, excluding non-controlling interests). The agent still treats the report as the official source
  (prompt rule 10) and does not know this difference. Debt also differs: yfinance `total_debt` is borrowings + leases
  (TCS and Infosys: leases only), while Wipro's report ratio (0.19) uses borrowings only (yfinance-style 0.23).
- **Currency bug in `get_financials` (found 2026-10-29, not fixed):** for `HCLTECH`, Yahoo labels `financialCurrency`
  as USD but the values are rupees, so revenue comes out about 100x too large (11,496,351.66 "crore"). Infosys is
  genuinely in USD and converts correctly, but its balance-sheet items use the *average* rate, not the year-end rate
  (equity 86,445 vs 92,852 in the report).
- **Eval scoring caveats (Milestone 5):** the tolerant score is a second, separate number; the strict `score_numbers` was
  not changed after seeing results. The 27 text grades were drafted by Claude Code and then reviewed (27 of 27 agreed),
  so they are not independent of its judgement; hedged answers (q15-q17 graph_full) and the refusals (q19, q20) were
  the judgement calls. The first full run used 1 repeat, so one run per cell cannot show run-to-run variation; the
  3-repeat run is still unsummarized. Number scoring only checks that some extracted number is within tolerance of an
  accepted value, not that the answer uses it correctly.
- **Out-of-scope tickers (F06, not fixed):** `get_financials` accepts any ticker and the raw agent has no scope check, so
  it answered HCLTech with a revenue about 100x too high. The graph's planner refuses such questions in under 2 s.
  Plan: make the tool reject tickers other than TCS, INFY and WIPRO.
- **Token cost (one measurement):** every `search_reports` call returns up to 5 chunks of up to 1,600 chars (about
  2,000 tokens by chars / 4). The loop sends the whole message list on every round, so a tool result is paid for
  again in each later round, and the rules allow up to 2 extra searches per fact. The long `search_reports`
  docstring is also sent on every call. The loop prints total input/output tokens per question but not a cost, and
  nothing caps tokens (only the 8-round limit). Measured once: "Compare TCS's net margin in FY2025-26 with what its annual report says about margins" used
  19,352 input and 429 output tokens over 4 rounds (about 45 times more input than output, which fits the history being resent). The 1,600-char index has 3,314 chunks vs 1,839 for 3,200 chars, so building it cost more to embed too.
- **Replanner:** a change request costs one extra model call even when it only asks for new wording. Added steps are not scope-checked in code, only by
  `REPLANNER_PROMPT`. The `auto_approve` path never reaches it.

## 7. Run with Docker
The image holds only the code and libraries. Keys come from `.env` at run time and `data/` (PDFs, Chroma index,
cache, SQLite files) is mounted from your disk, so the index and saved runs survive container restarts. `data/` must
already exist on the host (build it first with `python -m scripts.build_index`). Windows PowerShell, from the project folder:

```powershell
# build the image (re-run after code changes; the pip layer is cached unless requirements.txt changes)
docker build -t company-research-agent .

# run it: keys from .env, data/ mounted, app on http://localhost:8501
docker run --rm -p 8501:8501 --env-file .env -v "${PWD}/data:/app/data" company-research-agent
```

- Stop with Ctrl+C. `--rm` removes the container, not `data/`.
- `--env-file` does not strip quotes: write `OPENAI_API_KEY=sk-...` without `"` in `.env`.
- Untested limit: SQLite files on a Windows bind mount (`checkpoints.sqlite`, `usage.sqlite`) may be slow or lock under Docker Desktop.
