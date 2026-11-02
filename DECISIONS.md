## 2026-10-20: Chunk size of 800 tokens with 100 overlap
- Options: 400 / 800 / 1200 tokens
- Chose: 800, because retrieval hit rate was highest (see evals/retrieval.md)
- Trade-off: larger chunks cost more tokens per query

## 2026-10-13: Why LangChain `ChatOpenAI` with hand-written loops instead of prebuilt agents.
- Context: I only knew LangChain and LangGraph, so all LLM calls use `ChatOpenAI` with `bind_tools`.
- Chose: write the tool loop by hand (no `create_agent` or `AgentExecutor`).
- Why: so I understand what prebuilt agents hide. That hand-written loop became the `raw_agent` baseline.

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
- Likely cause: 52,820 is profit excluding exceptional items (footnote on
  p. 69: legal claim provision, restructuring, new Labour Codes impact).
  The reported profit attributable to shareholders is 49,210 (p. 167, p. 35),
  which matches yfinance. Found 2026-10-29 while building the eval answer key.
- Result: the two numbers are two definitions, not an error. 19.8% margin is
  on the adjusted profit; the reported margin is 18.43%. The agent's handling
  (report both, report is official) is unchanged and does not yet know why.
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

## 2026-10-27: Strip ISO dates and times before extracting numbers
- Problem / context: A draft's "2.0 percentage points" passed `find_unsupported` although no calculate step produced it. The `get_financials` evidence contained `'fetched_at': '2026-10-26T11:02:00.003372+00:00'`, and a piece of that timestamp rounded to 2.0 at one decimal place. Found with `explain_support` on the saved run (F03 in `evals/FAILURES.md`).
- Options considered: (1) ignore only the `fetched_at` field; (2) strip all ISO dates and times in the extractor; (3) drop the rounding tolerance.
- Chose: strip all ISO dates and times (`2026-03-31`, `2026-10-26T11:02:00.003372+00:00`, `11:02:00`) before extracting numbers, before the FY-label strip runs.
- Why: timestamps are labels, not claims, and can show up in any tool output, so fixing one field would leave the same hole elsewhere. Dropping the rounding tolerance would reject correct drafts like 18.4% against 18.42926.
- Trade-off: any time-like text (e.g. "18:30") is ignored too. Small numbers can still match real figures by chance; a stricter rule (derived numbers must come from `calculate`) is held back unless this recurs.
- Result (if measured): regression test with the exact timestamp now reports 2.0 as unsupported; 43 tests pass.

## 2026-10-27: Critic accepts drafts that honestly state evidence limits
- Problem / context: For "Who is the CEO of Wipro?", the critic loop spent 2 revisions (about 40 seconds and half the tokens) on missing evidence, which a rewrite can't fix. Found by the LangSmith trace: 3 writer and 3 critic runs, critic_passed false every time, and one critic model call of 23 s. The executor searched 4 times but never retrieved a passage stating the CEO in a sentence (p. 19 and p. 103 have one).
- Options considered: (a) lower MAX_REVISIONS to 1; (b) add a "fixable" field to the critic's verdict and send unfixable problems straight to human review; (c) route evidence gaps back to the executor for more searching; (d) change the critic prompt to accept drafts that honestly state the evidence's limits, and flag only problems a rewrite can fix.
- Chose: (d), the simplest change. (b) is the stronger fix if this recurs.
- Why: a rewrite can't create evidence that was never retrieved, so failing the draft only burns revisions, time and tokens.
- Trade-off: the critic may now pass a draft that says "not found" when better retrieval could have found the answer; that is a retrieval problem, tracked separately (hybrid search idea). In the after-run, the answer came from the Board of Directors list rather than a sentence, and the critic accepted it: a list row pairing a name and title is more reliable than a scrambled signature block, but the critic is now less strict about the names rule.
- Result: before 64.4 s, 54.6K tokens, $0.0062, 4 searches, 3 writer + 3 critic runs (2 revisions, limit hit), critic_passed false. After 30.5 s, 27.4K tokens, $0.0033, 2 searches, 2 writer + 2 critic runs (1 revision), critic_passed true. Answer: Srinivas Pallia, CEO & Managing Director, from the Board of Directors list.

## 2026-10-27: LangGraph instead of the raw loop
- Problem / context: The raw agent is one model in an 8-round loop that does planning, searching, writing and self-checking in one conversation. Failures F01 and F02 came from this: nothing checked the answer after the model wrote it, and there was no place for a human to stop a bad answer.
- Options considered: (a) keep the raw loop and add a check at the end; (b) LangGraph with separate nodes (planner, executor, writer, critic, human review).
- Chose: (b). The raw agent stays in raw_agent/ as the baseline.
- Why: I built both versions. With nodes, I could test each step alone (planner, then executor, then writer and critic). The checkpointer made pause, resume and restart possible, which the raw loop couldn't do. LangSmith shows each node separately, which is how I found the wasteful critic loop. A critic and a human approval step need control flow (loops back, pauses) that is explicit in a graph and messy in one loop.
- Trade-off: more code and more model calls per question. The Wipro CEO run took 64 s before the critic fix; the raw loop has no critic call. F04 also showed that rules in the old prompt did not carry over to the new node prompts.
- Result (if measured): none; raw vs graph not compared on the same questions yet.

## 2026-10-27: The critic has a code check and an LLM check
- Problem / context: The writer can put in numbers or facts the evidence does not support. F03 and F04 showed that each kind of check misses something the other catches.
- Options considered: (a) LLM check only; (b) code check only; (c) both in one node.
- Chose: (c). The code check requires every number in the draft to appear in the evidence (find_unsupported). The LLM check covers names, facts, citations, advice and the disclaimer.
- Why: numbers can be checked exactly, so a model should not guess. Names, roles and "does this page say this" cannot be checked by matching. F04 got through because the code check covers numbers only and the LLM check missed a signature block.
- Trade-off: the code check is loose (F03: a timestamp matched 2.0; small numbers can match by chance). The LLM check is slow (one call took 23 s in the trace) and is only as strict as its prompt.
- Result (if measured): F03 fixed with a regression test (43 tests pass); F04 re-test: critic flagged table/signature-only evidence.

## 2026-10-27: Revision limit of 2, and what happens at the limit
- Problem / context: Without a limit, a draft the critic keeps failing would loop forever.
- Options considered: (a) limit 1; (b) limit 2; (c) no limit, rely on the recursion limit.
- Chose: (b), MAX_REVISIONS = 2 in build.py. At the limit the run goes to human_review (or straight to finalize with auto_approve), and finalize adds a note listing the problems the critic still had.
- Why: Most fixable problems (a wrong number, a missing citation) are fixed in one or two rewrites; a third rarely helps and costs another writer + critic round. 2 balances quality against time and tokens.
- Trade-off: in the Wipro CEO trace the limit was reached with problems no rewrite could fix, costing about 40 s and half the tokens (see "Critic accepts drafts that honestly state evidence limits"). The limit bounds the cost but does not avoid it.
- Result (if measured): before the critic fix, 3 writer + 3 critic runs with the limit hit; after, 1 revision and critic_passed true.

## 2026-10-27: interrupt() at the top of human_review
- Problem / context: The human must see the draft and approve or ask for changes before the report is final. LangGraph resumes a paused node by running it again from its first line.
- Options considered: (a) interrupt() as the first thing in human_review; (b) interrupt() after other work in the node.
- Chose: (a). The node only calls interrupt() with the draft, critic_passed and critique, then returns human_feedback.
- Why: when the node re-runs on resume, anything above interrupt() runs twice. With nothing above it, the re-run is harmless. The pause inside the node lets me pass the draft out and the decision back in one call.
- Trade-off: the node must stay free of side effects (no model calls, no writes) before the interrupt. On "revise" it also clears critique so the writer does not get old critic problems with the new feedback.
- Result (if measured): none.

## 2026-10-27: SQLite checkpointer instead of in-memory
- Problem / context: The in-memory checkpointer lost a paused run when the program exited, so a draft waiting for human review could not be resumed later.
- Options considered: (a) InMemorySaver; (b) SqliteSaver at data/checkpoints.sqlite; (c) a server database such as Postgres.
- Chose: (b). build_graph(checkpointer=None) defaults to it; run.py prints the thread_id and has --resume <thread_id>.
- Why: a file next to the project keeps state across restarts with no server. The SqliteSaver docs show the sqlite3.connect(..., check_same_thread=False) pattern, which I used.
- Trade-off: SqliteSaver is meant for light, single-process use; checkpoints pile up in the file and nothing deletes old threads. Postgres would be the next step if this ever served several users.
- Result (if measured): full test on 2026-10-27. I started "Who is the CEO of Wipro?", stopped it at the approval prompt with Ctrl+C, and ran --resume a359a65b-cdda-470c-bceb-310abfd1df8f. It showed the same draft with no new tool calls, then finished after I approved.

## 2026-10-28: One flat TypedDict state; only `evidence` appends
- Problem / context: The nodes share data only through state, so I had to decide what goes in it and how a node's returned keys are merged.
- Options considered: (a) a flat TypedDict (ResearchState) with one key per piece of data; (b) nested state with a sub-dict per node; (c) a messages list like the raw loop.
- Chose: (a). Every key is overwritten when a node returns it; `evidence` is the only key with a reducer (operator.add, see the 2026-10-26 entry).
- Why: each node reads and writes named keys, so I can test a node alone with a small dict. Overwrite is right for draft, critique and critic_passed, because the newest version replaces the old one.
- Trade-off: any node can overwrite any key. critique is replaced on each critic run, so earlier critiques are lost, and human_review clears it by hand on "revise".
- Result (if measured): none.

## 2026-10-28: auto_approve is handled in routing, not inside human_review
- Problem / context: Evals must run without a person at the approval prompt, but human_review calls interrupt(), which always pauses.
- Options considered: (a) an auto_approve check inside human_review that skips interrupt(); (b) after_critic routes straight to finalize when auto_approve is set.
- Chose: (b). In build.py, after_critic returns "finalize" if state["auto_approve"], once the critic has passed or the revision limit is reached.
- Why: human_review stays "only call interrupt() and return" (see the interrupt() entry), with no branch above the pause. Unattended runs never reach the node at all.
- Trade-off: the skip is in the routing function, so a reader of human_review alone won't see it. auto_approve is an input in state that normal runs leave unset. A draft that failed the critic at the limit is also finalized unseen, with the note listing the critic's problems.
- Result (if measured): none yet; evals will use it in Milestone 5.

## 2026-10-28: Streamlit instead of FastAPI with HTML/CSS/JS
- Problem / context: The agent needs a page where a person can ask, watch progress, approve and download.
- Options considered: (a) Streamlit; (b) FastAPI backend with an HTML/CSS/JS front end.
- Chose: (a). FastAPI is listed as future work.
- Why: faster to build, and the agent is the project, not the UI.
- Trade-off: Streamlit re-runs the whole script on every click, so state handling is awkward (see the next entry), and the layout is less controllable. No API for other programs to call.
- Result (if measured): none.

## 2026-10-28: Only thread_id in session_state; the rest is read from the checkpointer
- Problem / context: Streamlit re-runs the script on every click, and a paused run must survive a refresh or a restart.
- Options considered: (a) keep plan, evidence and draft in st.session_state; (b) keep only thread_id and call graph.get_state() on every rerun; (c) thread_id also in the URL.
- Chose: (b) and (c). The thread_id is kept in the URL (?thread=...); the page reads the state from the SQLite checkpointer each rerun.
- Why: the checkpointer is already the single source of truth, so a second copy in session_state could go out of date. The URL survives a refresh, and session_state does not.
- Trade-off: one database read per rerun. Anyone with the URL can open the run, and there is no list of past runs.
- Result (if measured): both recovery cases tested: a refresh and a Streamlit restart each brought back the paused run.

## 2026-10-28: Model output is never rendered as HTML
- Problem / context: The draft and final report come from a model that reads retrieved text, so it could output HTML or script tags.
- Options considered: (a) st.markdown(..., unsafe_allow_html=True) for nicer formatting; (b) st.markdown without it.
- Chose: (b).
- Why: with HTML off, tags are displayed as text, so injected markup cannot run.
- Trade-off: no custom HTML styling in answers.
- Result (if measured): safety test passed: <script> and <img onerror> tags were shown as text and no pop-up appeared.

## 2026-10-28: Daily cap of 20 runs, counted in SQLite
- Problem / context: Each run costs OpenAI tokens, and the app could be left open or shared.
- Options considered: (a) no cap; (b) a counter in st.session_state; (c) a small SQLite table, data/usage.sqlite, with (date, count).
- Chose: (c), MAX_RUNS_PER_DAY = 20. The check runs before the graph starts; Approve and Send don't count.
- Why: session_state resets on refresh, so it can't enforce a daily limit. SQLite is already used for checkpoints and needs no server.
- Trade-off: one shared count for everyone, no per-user limit. The run is counted before it starts, so a failed run still uses one.
- Result (if measured): tested with the cap set to 1.

## 2026-10-28: Friendly errors; full traceback only in the terminal
- Problem / context: An exception during graph.invoke would show a full traceback in the page, which could include file paths or values from .env.
- Options considered: (a) let Streamlit show the exception; (b) catch it and show a short message with the exception type; (c) show the message too.
- Chose: (b). safe_invoke wraps every graph.invoke call, shows "Something went wrong while researching. Please try again." plus the error type, and prints the traceback to the terminal.
- Why: a user needs to know it failed, not see internals; I still need the traceback to debug.
- Trade-off: the error type alone doesn't say what went wrong, so I have to look at the terminal.
- Result (if measured): tested with a deliberate RuntimeError in the planner.

## 2026-10-28: Input checks before any run
- Problem / context: An empty or very long question would start a paid run that cannot give a useful answer.
- Options considered: (a) no check; (b) reject empty questions and questions over 500 characters before the run starts.
- Chose: (b). The checks come before the daily-cap check and before count_run().
- Why: rejected inputs cost no tokens and don't use up the daily cap, and they give the user a clear message instead of a confusing answer.
- Trade-off: 500 is a guess, not measured. Valid but strange input (e.g. a question about another company) is still handled by the planner's scope check.
- Result (if measured): none.

## 2026-10-29: Eval answer key built by hand from two sources
- Problem / context: Milestone 5 needs a correct answer for each of the 20
  questions, found without the agent, so the three agent versions can be scored.
- Options considered:
  1. Use yfinance values only.
  2. Use annual-report values only.
  3. Collect both, and accept either where they differ.
- Chose: Option 3. `accepted_values` holds the yfinance value and the report
  value where they differ (q03, q04, q06, q08; q09/q10 hold both Wipro
  definitions). Tolerance: 0.5% relative for rupee amounts, 0.1 percentage
  points for margins and growth, 0.02 absolute for debt-to-equity.
- PAT decisions: TCS 52,820 (excluding exceptional items) is not accepted;
  reported 49,210 is. Wipro accepts both 13,197.4 (equity holders) and
  13,265.5 (profit for the year incl. non-controlling interests).
- Risk questions: pass if the answer names at least 2 risks from the report's
  list and cites a page inside that risk section.
- Why: yfinance and the annual reports sometimes measure the same thing
  differently (USD conversion, which profit line), so accepting only one
  source would mark correct answers wrong. The 2-risk rule keeps the risk
  questions scoreable, because the risk lists are long and a short answer
  can't name them all.
- Trade-off: accepting two values makes scoring more lenient; the Infosys
  yfinance values are 0.3-0.6% off because of USD conversion; the risk rule
  does not check that the named risks are the most important ones.
- Result (if measured): not scored yet.

## 2026-11-01: Report strict and percentage-tolerant number scores, and disclose how text answers were graded

- Problem / context: In the milestone 5 run (2026-10-30_192104), graph_no_critic failed q09 and q10 under `score_numbers`. Its answers gave the debt-to-equity ratios as percentages ("22.9%", "10.5%", "9.9%"), while the answer key accepts ratios (0.23, 0.1, 0.099). The numbers match; only the format differs. Separately, q12-q20 can't be scored by code, so they were hand-graded.
- Options considered:
  1. Keep only the strict score and count the format mismatch as a failure.
  2. Change `score_numbers` so percentages are accepted.
  3. Leave `score_numbers` unchanged and add a second score in `summarize.py` that also accepts "X%" as X/100, then report both.
  For the text questions: grade them all by hand from scratch, or have Claude Code draft the grades and review every row.
- Chose: Option 3. The tolerant rule applies only to plain-ratio questions (q09, q10). The percentage-point questions (q07, q08, q11) stay strict. Both scores appear in the summary table. For q12-q20, Claude Code drafted the 27 grades from the pass rules in the answer key, and I reviewed all 27 rows (27 of 27 agreed) before copying them into the `pass` column.
- Why: A ratio of 0.229 and 22.9% are the same value, so failing it would measure formatting, not correctness. But changing the original scorer after seeing the results would look like moving the goalposts. Keeping the strict score and adding a separate tolerant one shows both, so a reader can see exactly how much of the gap is format and how much is wrong numbers. Points from the discussion: the strict score is the original, unchanged definition of correct. The tolerant score shows how much of the gap is formatting and how much is wrong numbers. Reporting both keeps the comparison honest, because the two scores give different rankings for graph_no_critic.
- Trade-off: Two numbers per version are harder to read than one. The tolerant rule could pass a wrong answer that happens to match after dividing by 100, which is why it is limited to q09 and q10. The text grades were drafted by Claude Code, which also helped build parts of this project. Agreement on all 27 rows shows I accepted the drafts, not that they're independent of Claude's judgement. The q15-q17 graph_full hedges ("does not clearly state...") and the q19 and q20 refusals are the judgement calls.
- Result (if measured): Numbers, graph_no_critic: 9/11 strict, 11/11 tolerant. Overall, graph_no_critic: 18/20 strict, 20/20 tolerant. raw: 19/20 on both scores (its only fail is q19, which gave an HCLTech revenue number). graph_full: 20/20 on both. The tolerant score is the only thing that changes the ranking between graph_no_critic and graph_full.

## 2026-11-02: Settings moved to config.py
- Problem / context: model names, round and revision limits, chunk size, collection name and daily cap were scattered across files, and chunker, store and build script disagreed on defaults.
- Options considered: leave as is; one config.py with .env overrides.
- Chose: config.py, with defaults equal to the values used in the evals. Model names have no default: they are read from .env when first used.
- Why: one place to change; the index and eval defaults can't drift apart.
- Trade-off: a missing model name fails with a clear error instead of a KeyError.

## 2026-11-02: Send change requests through a replanner node

- Problem / context: In human review, "Request changes" sent the run straight back to the writer. The writer can only use the evidence already collected, so a request that needed a new fact (for example "who is the CEO?") could not be answered (F07).
- Options considered:
  1. Keep sending feedback to the writer and tell the user to ask a new question.
  2. Send the feedback back to the planner and rerun the whole plan.
  3. Add a replanner node that turns the feedback into only the extra steps needed, then sends the run to the executor or the writer.
- Chose: Option 3. `human_review` routes to `replanner` when there is feedback, and to `finalize` on approve. The replanner makes one structured-output call with the question, the steps already done and the change request. It returns a list of new steps, appended to `plan`, with `current_step` set to the old plan length. `after_replanner` then routes:
  - new steps go to the executor, then the writer, critic and review again;
  - an empty list (wording-only or out-of-scope request) goes straight to the writer.
- Why: Only the replanner can add steps. It reuses the existing executor, writer and critic. Executing only the new steps keeps finished work and does not repeat tool calls. The re-test found Salil Parekh, p. 14.
- Trade-off: Every change request now makes an extra model call, even for "make it shorter". The scope check for new steps is only in the replanner prompt. Both after_replanner branches are tested (tests/test_routing.py, 17 routing tests).
- Result (if measured): one re-test (F07): the request now fetched the CEO's name. Not measured on cost or latency.

## 2026-11-03: Run the app in Docker (Python 3.14 slim, data/ as a volume, keys at run time)

- Problem / context: The app only ran on my machine, inside my own venv and with my own .env. I wanted one command that runs it anywhere without baking secrets or the 1600-character index into the image.
- Options considered:
  1. Copy .env and data/ into the image so it runs with no arguments.
  2. Use a slim image and keep data/ and the keys outside it: mount data/ as a volume and pass the keys with `--env-file` at run time.
- Chose: Option 2. The Dockerfile starts from `python:3.14-slim`, installs requirements.txt, copies the code and starts Streamlit on port 8501. data/ is mounted as a volume. The keys come in at `docker run` time.
- Why: The image holds no secrets, so it is safe to share. data/ holds the Chroma index and the SQLite checkpointer, so a volume means the index is not rebuilt and a paused run is not lost when the container is replaced.
- Trade-off: The run command is longer (`--env-file` plus a volume). Python 3.14 is new, so some packages may not have wheels yet. `--env-file` keeps quotes as part of the value, so .env must have no quotes (F09).
- Result (if measured): build 138 s; image holds no .env or data/; app ran in the container after the quote fix (F09); a paused run survived a container restart.
