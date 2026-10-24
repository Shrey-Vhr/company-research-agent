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