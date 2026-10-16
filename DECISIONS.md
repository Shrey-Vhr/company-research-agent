## 2026-10-20: Chunk size of 800 tokens with 100 overlap
- Options: 400 / 800 / 1200 tokens
- Chose: 800, because retrieval hit rate was highest (see evals/retrieval.md)
- Trade-off: larger chunks cost more tokens per query

## - - Why LangChain `ChatOpenAI` with hand-written loops instead of prebuilt agents.
- Why Streamlit for the UI (section 3).

## Explore the data. For TCS.NS, INFY.NS and WIPRO.NS, print the rows you need. Check one revenue number against that company's annual report. Write any differences in DECISIONS.md.

## 2026-10-14: One `calculate(expression)` tool instead of `ratio` + `percent_change`
- Options: (a) one general `calculate` tool that parses arithmetic with `ast`; (b) separate `ratio(a, b)` and `percent_change(old, new)` tools
- Chose: (a), because one tool covers margins, growth, debt-to-equity, sums and averages without adding a new tool
  for every metric. Fewer tools also means fewer chances for the LLM to pick the wrong one.
- Safety: never `eval()`. A character allowlist runs first, then `ast.parse(mode="eval")` and my own tree walk that only
  accepts numbers, `+ - * /` and unary `+ -`. Everything else (`**`, `//`, names, calls, attributes) is rejected.
  A 200-character limit stops very large or deeply nested input.
- Trade-off: the LLM must write the formula correctly (e.g. growth = (new - old) / old * 100), which named tools would
  do for it. No `**`, so CAGR is not possible yet. Add `**` with a size limit if CAGR is needed later.

## 2026-10-14: Checked yfinance revenue against the TCS annual report
- Checked: TCS.NS `income_stmt.loc["Total Revenue"]`, year ended 2025-03-31 = 2,553,240,000,000 INR
- Annual report FY2024-25 (Directors' Report, consolidated revenue from operations) = ₹2,55,324 crore
- Result: exact match (2,553,240,000,000 / 10^7 = 2,55,324 crore). FY2024 also matches: ₹2,40,893 crore.
- Differences to handle (these are about format, not wrong numbers):
  - Units: yfinance gives raw rupees, the report uses crore (1 crore = 10^7). Convert before showing or comparing.
  - Scope: yfinance is the **consolidated** figure. The standalone revenue in the same report is ₹2,14,853 crore,
    which is about 16% lower. Always compare against the consolidated statements.
  - Revenue only: "Total Revenue" equals "Operating Revenue", so it does not include other income.
  - Labels: columns are fiscal year-end dates (2025-03-31 = FY2024-25), not calendar years.
  - History: yfinance only returns the last 4 years.
- Chose: use yfinance consolidated revenue as-is, convert to ₹ crore for display, label years as FY.
- Not checked yet: INFY.NS and WIPRO.NS, and any row other than revenue.

## 2026-10-17: parallel_tool_calls=True in the raw agent loop
- Problem / context: Comparison questions ("Compare TCS and Wipro margins") need one get_financials call per
  company. With parallel_tool_calls=False the model asks for one tool per reply, so each company costs a full
  extra round, and every round resends the whole message history to the model.
- Options considered: (a) False: one tool call per model reply (what playground/loop.py used);
  (b) True: the model may return several tool calls in one reply.
- Chose: (b) True.
- Why: the comparison test "Compare Infosys and Wipro revenue growth over the
  last 3 years" hit the 8-round limit with no answer. With False, each of the
  6 independent calculate calls took its own round (2 fetches + 6 calcs = 8),
  leaving no round for the final answer.
- Trade-off: "parallel" only means several calls in one reply. The loop still runs them one by one, so tool
  time is not saved. The model can also send a calculate call in the same reply as the get_financials calls it
  depends on, before it has the real numbers, which breaks rule 1 (every figure from a tool). Every call needs
  its own ToolMessage with the matching tool_call_id, which the loop already does.
- Result (if measured): before (False): 8 rounds, stopped without an answer, twice.
  After (True): [I will fill this in after rerunning].
