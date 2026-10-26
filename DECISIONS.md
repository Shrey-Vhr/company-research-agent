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

## 2026-10-18: Filter search by company and FY metadata
- Problem / context: An unfiltered search mixes all three companies: "number of employees" returned
  WIPRO, TCS, WIPRO, TCS, INFY. A question about one company needs chunks from that company only.
- Options considered: (a) no filter, let the model sort out which company each chunk is from;
  (b) search more chunks and drop other companies afterwards in Python; (c) Chroma metadata filter
  (where) on company and fy, applied inside the search.
- Chose: (c) optional company and fy arguments; one condition is passed directly, two are combined with $and.
- Why: the filter runs before ranking, so all k results come from the right company instead of k
  minus the ones thrown away; the metadata is already on every chunk from the loader.
- Trade-off: the filter value must match exactly ("INFY", not "Infosys"; "FY2025-26"), and a wrong
  value returns [] with no error. Comparing two companies needs two searches.
- Result (if measured): "number of employees" with company="TCS" returned 5 TCS chunks, top hit
  p. 108 (employee table); with company="INFY", fy="FY2025-26" 5 INFY chunks; a wrong name ("Infosys") returned [].

## 2026-10-18: Search returns top 5 chunks as plain dicts with distance score
- Problem / context: search results go to the agent (and later a tool), which needs text plus
  citation fields, not LangChain Document objects.
- Options considered: k = 4 (Chroma default) or 5; return Documents, or dicts with selected fields;
  with or without a score cutoff.
- Chose: k=5 by default; return dicts text, company, fy, pdf_page, score (rounded to 3 decimals); no cutoff yet.
- Why: dicts are easy to print, test and turn into a tool result; pdf_page is the number we cite;
  the score is kept so a cutoff can be chosen later from real numbers.
- Trade-off: always returns 5 chunks, even weak ones (INFY "number of employees" best match is about
  customers, p. 143). 5 chunks of up to 3,200 chars is about 4,000 tokens per search sent to the model.
  The score is an L2 distance (lower = closer), not a percentage, and scores bunch together (about 1.0–1.25).
- Result (if measured): 3 test searches in about 9 s in total, including startup and one embedding call each.

## 2026-10-19: Build the retrieval answer key with plain text search
- Problem / context: The retrieval eval needs gold pages for each question, found without using the
  system being tested.
- Options considered: (a) label pages from my own vector search results; (b) regex/keyword search
  over the raw PyMuPDF page text (page index + 1, same numbering as the loader).
- Chose: (b)
- Why: the answer key stays independent of the system it tests; labels from vector search would
  only reward the search for agreeing with itself.
- Trade-off: manual and slower; tables extract as runs of numbers ("1 126 127 ... 185 185"), so
  snippets are less readable.
- Result (if measured): all 6 answers found; it also caught a wrong old note
  ("also on p. 12" for r01; p.12 has no headcount).

## 2026-10-19: Fix eval questions so each one has a single clear answer in its own company's report
- Problem / context: In r02–r05 the `company` field named a different company than the question
  (e.g. company TCS, question about INFY's dividend). No report mentions a competitor's answer, so
  with the company filter these questions could never succeed. r04 (offices: 311 vs 312) and
  r05 ("a named risk": ~10 valid answers) were ambiguous.
- Options considered: keep the questions as cross-company "trap" questions; or match each
  `company` to the question and reword the ambiguous ones.
- Chose: matched every `company` to the company the question asks about; r01 -> "permanent
  employees at year end"; r04 -> "international offices" (185); r05 -> "How does Infosys describe
  cybersecurity as a business risk?" (p.134).
- Why: a retrieval eval should test whether search finds the page, not whether the question is
  answerable at all; one clear answer per question makes a hit or miss unambiguous.
- Trade-off: no question currently tests cross-company or unanswerable cases.
- Result (if measured): 6 questions, each with a single answer that can be checked by page.

## 2026-10-19: answer_pages is a list of every page that fully answers the question
- Problem / context: Most answers appear on several pages (r03 on 8 pages). With one gold page,
  a search that returns another correct page would be scored as a miss.
- Options considered: one gold page per question; or a list of acceptable pages.
- Chose: `answer_pages` list; a page is included only if it fully answers the question, and the
  `note` says why near-misses were left out.
- Why: p.108 left out for r01 because it gives 6,17,437 including non-permanent staff; p.243
  left out for r02 because it only mentions the final Rs 25, not the total; p.42, p.108, p.110,
  p.132 left out for r05 because they mention cybersecurity but do not describe it as a business risk.
- Trade-off: lists are uneven (r04 and r05 have 1 page, r03 has 8), so questions with many copies
  of the answer are easier to hit; deciding "fully answers" is a judgement call.
- Result (if measured): not measured yet (no scoring script).

## 2026-10-20: Retrieval eval scores by page, filtered by company, top 5
- Problem / context: Need a repeatable number for how well search() finds the answer page.
- Options considered: score by page vs by answer text; filter by company vs search all reports; k = 3/5/10.
- Chose: search(question, company=item["company"], k=5); compare result pdf_pages with answer_pages via is_hit;
  report total and per-company hit rate; save each run to evals/results/retrieval_<date>.json.
- Why: k=5 is what the agent will get; the company filter matches how the agent will search;
  saved runs let me compare before/after a change (chunk size, k, embeddings).
- Trade-off: a page hit doesn't prove the chunk contains the answer (long pages are split into chunks);
  6 questions is too few for small differences to mean anything. Questions with more accepted pages are
  easier to hit (r03 has 8, r04 has 1), so per-company hit rates are not fully comparable.
- Result (if measured):  baseline with 3,200-char chunks: 3/6 (50%). Hits on descriptive
  questions (risk, acquisition) and r01; misses on short exact facts
  (dividend, CEO name, office count).

## 2026-10-24: Chunk size 1,600 chars (overlap 200) instead of 3,200 (overlap 400)
- Problem / context: Retrieval eval hit rate was 3/6 with 3,200-char chunks (1,839 chunks). Large chunks hold
  many topics, so one chunk's embedding is a blur and the right page can rank below others.
- Options considered: 3,200/400 (current) vs 1,600/200, each in its own Chroma collection
  (annual_reports vs annual_reports_1600), same questions, k=5, company filter.
- Chose: 1,600/200 as the new default (3,314 chunks). The 3,200 collection stays on disk for comparison.
- Why: hit rate rose from 3/6 to 4/6. The only question that changed was r03 (WIPRO CEO): MISS -> HIT
  (p.145). r01, r05, r06 hit on both; r02 and r04 miss on both. On r05 the right page (134) moved from 2nd
  to 1st, and the top score was lower (closer) on r04 and r06 (r05 about the same).
- Trade-off: 80% more chunks (3,314 vs 1,839), so more embedding cost and ~5 chunks of ~1,600 chars sent to the
  model instead of ~3,200. The result is small: one question out of six is a small difference. It points
  clearly in one direction, but 6 questions can't prove it. That's why the plan calls for 15 questions.
  Also, r03 has 8 accepted pages, so it is the easiest to hit, and r02 and r04 still miss with both sizes,
  so chunk size is not what is failing there.
- Result (if measured): 3/6 vs 4/6 (TCS 1/2 vs 1/2, INFY 1/2 vs 1/2, WIPRO 1/2 vs 2/2). Both runs
  are in evals/results/ (retrieval_annual_reports_*.json, retrieval_annual_reports_1600_*.json).

## 2026-10-25: Net profit differs between yfinance and the TCS annual report
- Problem / context: For "Compare TCS's net margin with what its annual report
  says", the two sources gave different profit figures for FY2025-26.
- Figures: yfinance net income ₹49,210 crore (margin 18.4%); annual report
  profit after tax ₹52,820 crore (p. 69) and PAT / total revenue 19.8% (p. 70).
- Checked: both report numbers appear in the cited passages (text check);
  ₹52,820 crore is in the [FY2026 / FY2025] column (checked by eye on p. 69).
- Likely cause: Not yet identified; the report’s definition needs checking.
- Handling: the agent reports both figures, gives the gap, and treats the
  annual report as the official source (prompt rule 10).
- Trade-off: answers show two numbers instead of one, which is less tidy but
  honest; garbled TCS table headers mean the year column must be checked by eye.

## 2026-10-26: `evidence` uses an append reducer, and each item keeps the raw tool output
- Problem / context: In the LangGraph state, the executor visits one plan step at a time and each visit finds new facts. A LangGraph node returns only the keys it changed, and by default a returned key overwrites the old value. With a plain `list`, every executor step would replace the evidence from earlier steps. The critic also needs to check each claim in the draft against the numbers the tools returned. It can't do that if evidence is only a summary.
- Options considered:
  1. Plain `list[dict]` with no reducer. Each node reads the old list, appends, and returns the whole list.
  2. `Annotated[list[dict], operator.add]`. Each node returns only its new items and LangGraph concatenates them onto the existing list.
  3. Store only a short summary or the extracted fact per item.
  4. Store each item as `{step, tool, citation, content}`, where `content` is the raw tool output.
- Chose: Options 2 and 4 together. `evidence: Annotated[list[dict], operator.add]`, and each item holds `step`, `tool`, `citation` and the unmodified `content` text from the tool.
- Why:
  - The reducer makes appending the default. A node returns `{"evidence": [new_item]}` and the old items stay. Option 1 depends on every node remembering to copy the old list, and one forgotten copy silently wipes earlier evidence.
  - Raw `content` lets the critic verify claims in code instead of asking an LLM. For a claim like "TCS revenue was X", plain Python can check that the number appears in the `content` of an item with the matching `tool` and `citation`. A summary written by the model can't be checked this way, because a hallucinated summary would pass its own check.
  - `step` and `citation` tie each fact back to the plan step that produced it and to its source, which the report needs for grounded citations.
- Trade-off:
  - Raw outputs are long, so evidence takes more memory and state size. If it is ever passed to a model, it costs more tokens, so I may need to trim what goes into prompts.
  - `operator.add` can only append. A node can't remove or correct an item. If a retry step produces duplicate or bad evidence, both copies stay, and a custom reducer would be needed to dedupe.
  - Code-based checks are only as good as the matching. A number can be formatted differently in the claim and in the raw text. Seen in this project: TCS headcount appears as 5,84,519 and 584,519, and the TCS PDF splits digits (1 58 for 158). The critic needs a normalising step.
- Result (if measured): not measured yet.
