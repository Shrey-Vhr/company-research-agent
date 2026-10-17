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

## 2026-10-17: Annual reports for the RAG corpus
- Problem / context: The agent needs official documents to retrieve from and cite, alongside the yfinance numbers.
- Options considered: (a) FY2025-26 reports of TCS, Infosys, Wipro;
  (b) two years of reports per company (FY2024-25 and FY2025-26);
  (c) companies from different sectors (e.g. a bank and an IT company);
  (d) quarterly results instead of annual reports.
  Chose (a): one year keeps the index small; one sector makes comparisons fair;
  annual reports have the most text to retrieve from.
- Chose: FY2025-26 annual reports of TCS, Infosys and Wipro (files and source URLs in docs/DATA_SOURCES.md).
- Why: same FY as the latest yfinance data, all three are IT services companies so comparisons make sense,
  and they're official and citable.
- Trade-off: Only one year of reports, so questions about earlier years rely on yfinance alone. The PDFs are
  large (360 / 383 / 548 pages), which means more chunks to embed and search.
- Result (if measured): TCS consolidated revenue from operations FY2025-26 is
  ₹2,67,021 crore in the annual report (PDF p. [n]); yfinance returns the same.
  Infosys and Wipro not checked yet.

## 2026-10-17: Which page number to cite from annual reports
- Problem / context: PyMuPDFLoader numbers pages from 0, and the number printed on a report page differs
  from the PDF page (TCS PDF p. 181 = printed p. 211). Citations need one number a reader can check.
- Options considered: (a) printed page number parsed from the text; (b) PDF page, 1-based (loader page + 1).
- Chose: (b) store and cite pdf_page = metadata["page"] + 1.
- Why: the PDFs have no page labels, many PDF pages are two-page spreads with two printed numbers,
  and extracted digits are sometimes split ("1 31"), so the printed number can't be read reliably.
  The PDF page is always there and is what you type into a viewer to verify.
- Trade-off: it doesn't match the number printed in the report or its table of contents, so a reader
  holding a printed copy has to use the PDF viewer's page box.
- Result (if measured): 1,286 of 1,291 pages kept; offset is about +30 (TCS 181→211, Infosys 192→223, Wipro 274→300).

## 2026-10-17: Chunk size and overlap for annual-report chunks
- Problem / context: Report pages must be split into chunks before embedding. Chunks that are too small
  separate a number from its label; chunks that are too big mix unrelated topics.
- Options considered: chunk_size 3200 / chunk_overlap 400 characters, using
  RecursiveCharacterTextSplitter with its default separators (paragraph, line, word).
- Chose: chunk_size=3200, chunk_overlap=400 (characters, not tokens).
- Why: a report page is a median of 2,812 characters, so 3200 keeps most pages (61%) as one whole
  chunk, and a table or section stays together with its headings. 3200 characters is roughly
  800 tokens, small enough to send several retrieved chunks to the model. 400 overlap (12.5%) means
  a sentence or table row cut at a chunk boundary still appears whole in one of the two chunks.
- Trade-off: big chunks dilute the embedding, so a search for one figure matches a whole page of
  mixed rows and is less precise. More text per chunk also means more tokens per answer.
  Overlap repeats text, adding about 12% more to embed and store.
- Result (if measured): 1,286 pages -> 1,839 chunks (median 2,402 chars, smallest 93);
  779 of 1,286 pages fit in a single chunk. Retrieval quality not measured yet.

## 2026-10-17: PDF loader for annual reports (TCS broken text)
- Problem / context: TCS text has broken words and split numbers ("C onsol i d ated", "2 0 2 5- 2 6", "1 58").
- Options considered: (a) PyMuPDF (PyMuPDFLoader, current); (b) pypdf (PyPDFLoader); (c) regex cleanup
  of the extracted text.
- Chose: keep PyMuPDF; no cleanup yet; re-check with TCS retrieval scores in step 7.
- Why: pypdf breaks the same words in the same places, so the problem is in the TCS PDF, not the loader.
  pypdf also glues text together ("ReportsCorporate", "2 61 58") and is much slower. Regex cleanup is
  risky without a measure of harm: it could merge real words or table numbers, so wait until step 7
  shows whether the broken text actually hurts TCS retrieval.
- Trade-off: TCS chunks keep the broken text for now, so searching for words like "Consolidated" or
  "Vulnerable" may miss TCS pages, and split numbers like "1 58" won't match "158".
- Result (if measured): broken score PyMuPDF 50 vs pypdf 54 (TCS pdf pages 128, 150, 181; Infosys 192;
  Wipro 274, via playground/compare_loaders.py); Infosys and Wipro clean with both.

## 2026-10-17: Batch size when adding chunks to Chroma
- Problem / context: 1,839 chunks must be embedded; one OpenAI embeddings request is limited to 300,000 tokens.
- Options considered: (a) send all chunks in one add_documents call; (b) fixed batches of 100 chunks.
- Chose: (b) BATCH_SIZE = 100.
- Why: 100 chunks x 3,200 chars is about 80,000 tokens, well under the limit, and a failure loses at most one batch;
  fixed IDs mean re-running after a failure just overwrites what was saved.
- Trade-off: 19 calls instead of 1, so slightly slower; the 4-chars-per-token estimate is rough for number-heavy pages.
- Result (if measured): build_index embedding ended with a count of 1839 two times and it ran in approx. 37 seconds.
