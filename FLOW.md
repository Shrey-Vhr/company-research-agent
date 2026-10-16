# FLOW

## 1. What the project does
A research agent that answers questions about NSE-listed Indian companies (e.g. TCS, Infosys, Wipro).
An OpenAI model, via LangChain, calls tools to fetch yearly financials from Yahoo Finance and do arithmetic.
Answers cite the source and financial year, never give investment advice, and end with a disclaimer.

## 2. Current status
- **Stage:** hand-written tool-calling loop (`raw_agent`), before LangGraph, RAG, UI and evals.
- **Works:** two tools (`get_financials`, `calculate`), a system prompt, the loop with an 8-round limit,
  a command-line entry point, a 7-day JSON cache, and 22 offline unit tests for the tools.
- **Next (planned, from requirements.txt):** annual-report RAG (PyMuPDF + Chroma), a LangGraph version
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
| `research_agent/config.py` | Empty (placeholder) |
| `research_agent/**/__init__.py` | Empty package markers |
| `tests/test_calculator.py` | Valid/decimal/divide-by-zero/malicious-input tests for the calculator |
| `tests/test_financials.py` | Parsing, missing-row and invalid-ticker tests using a fake `yf.Ticker`; no network |
| `tests/fixtures/tcs_sample.json` | Two years of TCS statements in raw rupees, shaped like yfinance |
| `playground/*.py` | Learning experiments (yfinance rows, a first tool-calling script); not used by the app |
| `data/cache/` | Cached `get_financials` results, one JSON per ticker (gitignored) |
| `.env` | `OPENAI_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, LangSmith settings (gitignored) |
| `requirements.txt` | Dependencies, including ones for later stages (LangGraph, Chroma, Streamlit) |
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
- **No tests for `run_agent`** (they would need a fake model). Only the tools are tested.
- **Not cross-checked:** only TCS revenue has been checked against an annual report; Infosys/Wipro and the other fields haven't.
