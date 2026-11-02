## F01: Wrong CEO name from a signature block (2026-10-24)
- Question: "Who is the CEO of Wipro, according to its annual report?"
- Wrong answer: Anand Subramanian (actually the auditor partner)
- Correct answer: Srinivas Pallia (Wipro AR FY2025-26, p. 103)
- Cause: signature block on p. 184; extraction put the CEO title above the
  auditor's name, and the model trusted the order.
- Fix: prompt rule: names and roles only from sentences, not tables or signature blocks.
- Result after fix: Correct name, from a sentence passage (p. 19), not the signature block.

## F02: Answer from model's own knowledge with an unsupported citation (2026-10-24)
- Question: "Who is the CEO of Wipro, according to its annual report?"
- Answer: Srinivas Pallia (correct), cited to p. 184
- Problem: none of the 5 retrieved passages contained "Pallia"; the model used
  its training knowledge and cited a passage that doesn't support the claim.
- Why it matters: for an unfamiliar fact, it would invent an answer with a fake citation.
- Fix: grounding rules in the system prompt (facts only from tool results,
  citation must contain the fact, retry with new wording, else say not found).
- Long-term fix: the Milestone 3 critic's code check (every name and number
  in the draft must appear in the evidence).
- Result after fix: Re-test cited p. 19; that passage contains ‘Pallia’ (verified with a text check).

## F03: Code check passed a number the writer computed itself (2026-10-27)
- Draft said "2.0 percentage points"; no calculate step produced it.
- Cause: extract_numbers read the cache timestamp in get_financials output
  ('fetched_at': ...T11:02:00.003...) as numbers, and one rounded to 2.0.
- Found by: freezing the run's evidence and testing the number check on it
  (explain_support), instead of re-running the model until it repeated.
- Fix: ignore ISO dates and times when extracting numbers; regression test added.
- Remaining risk: small numbers can still match real figures by chance;
  stricter rule (derived numbers must come from calculate) if it recurs.

## F04: F01 came back after the LangGraph rebuild (2026-10-27)
- Question: "Who is the CEO of Wipro?"; answer: Anand Subramanian (auditor), p. 184.
- Cause: the F01/F02 rules were in the Milestone 1 system prompt only; the
  new graph prompts (executor, writer, critic) never got them.
- Also missed by: the critic (code check covers numbers only; LLM check didn't
  spot the signature block) and by human review (approved without checking).
- Fix: names-from-sentences and retry rules ported to executor, writer and critic.
- Lesson: when rebuilding, re-test every old failure. Milestone 5 evals must
  include an F01-style question so this is caught automatically.
- result: “Re-test: correct name (Srinivas Pallia); writer flagged that evidence was table/signature only; critic flagged it too. Sentence passages (p. 19, 103) not retrieved: known retrieval limit.”

## F05: Debt-to-equity written as percentages; scorer marked it wrong (2026-11-01)
- Runs: q09 and q10, graph_no_critic (eval run 2026-10-30_192104)
- Answer: Wipro 22.9%, TCS 10.5% (evidence: 0.229, 0.105), plus a gap of 12.4
  points the writer computed itself
- Scored: fail, because the scorer compares 22.9 with 0.229
- Real status: values correct, different format; the gap figure broke the
  "every number from a tool" rule
- In graph_full, the critic's code check flagged 22.9, 10.5 and 12.4 (8 items,
  duplicates), and the rewrite used the evidence's format and passed
- Lessons: (1) the critic's measured gain here was format consistency plus
  one ungrounded figure, not factual correctness; (2) the scorer should
  accept percentage forms of ratios; (3) the writer prompt should keep
  numbers in the evidence's form; (4) remove duplicate critique items
- Repeat run (3x): graph_no_critic wrote percentages on q09 and q10 in 6 of 6
  runs; graph_full used ratios on q09 in 3 of 3 but percentages on q10 in 2 of
  3, so the critic's format fix is not reliable.

## F06: Raw agent answered an out-of-scope company with wrong data (2026-10-30)
- Run: q19, raw. Answer: HCLTech revenue ~₹1,14,96,351 crore (about 100x too high)
- Cause: get_financials accepts any ticker; Yahoo's currency label for
  HCLTech is wrong; the raw agent has no scope check
- Graph versions refused in under 2 s (planner scope check)
- Lesson: the clearest real advantage of the graph in this eval is safety on
  out-of-scope questions. Fix the tool to reject tickers outside TCS, INFY, WIPRO.
- Repeat run: raw gave the wrong HCLTech revenue in 3 of 3; both graph
  versions refused in 3 of 3.

- Fixed in v1.0: get_financials rejects tickers outside TCS, INFY, WIPRO.

## F07: Revise requests could not fetch new facts (2026-11-02)
- Fixed with a replanner node; re-test found Salil Parekh, p. 14.

## F08: Critic flagged a page number as an unsupported figure (2026-11-02)
- The critic flagged a page number from a pp. citation list as an unsupported figure.
- Fixed in extract_numbers; regression test added.

## F09: OpenAI 401 inside Docker from quoted .env values (2026-11-03)
- Where: the Docker container (docker run --env-file .env ...); the app showed
  "Something went wrong" with error type OpenAIAuthenticationError
- Error: 401 "Incorrect API key provided", and the masked key in the message
  started and ended with a quote mark ("sk-proj...eMA")
- Cause: .env had OPENAI_API_KEY="sk-proj-...". python-dotenv (local runs)
  strips the quotes, so it worked on my laptop; Docker's --env-file keeps
  them, so inside the container the key really began with a " character
- Fix: removed the quotes (and any spaces around =) from every .env value;
  added a comment to .env.example: "# No quotes around values: Docker's
  --env-file keeps them."
- Result: the same question ran in the container and still ran locally
- Lesson: "works on my machine" can hide config differences. Two tools that
  read the same .env file can read it differently, so test the container
  with the real config, not just the build.