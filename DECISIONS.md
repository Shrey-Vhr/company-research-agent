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
