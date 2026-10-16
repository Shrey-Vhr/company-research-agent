# Rules for this project
I am a student building this project to learn agentic AI. I must be able to
explain every line in an interview.

- Do NOT write or edit: the agent loop, system prompts, LangGraph state/nodes/
  edges/routing, the critic, retrieval design, eval questions or scoring,
  unless I explicitly say "write this for me".
- You MAY write data plumbing when I ask: tool function bodies (yfinance,
  pandas, parsing), the calculator parser, caching, PDF loading, Streamlit
  layout, Docker/CI. Keep each piece small.
- After writing any code for me, explain it in 5 plain lines: inputs, outputs,
  what happens on error. Then ask me one question to check I understood.

## Project documentation (do this automatically)

### DECISIONS.md
- Whenever we make a real choice (a design option, a library, a setting,
  a fix for a failure, a trade-off), draft an entry and show it to me before
  saving. Ask: "Add this to DECISIONS.md?" Save it only after I confirm or edit.
- Format:
  ## YYYY-MM-DD: <short title>
  - Problem / context:
  - Options considered:
  - Chose:
  - Why:
  - Trade-off:
  - Result (if measured):
- Never rewrite or delete existing entries. Add new ones at the bottom.
- Use my own wording when I give a reason.

### FLOW.md
- Keep FLOW.md up to date as the single description of how the project works
  right now. Update it yourself, without asking, after any change to code
  structure, files, tools, prompts or the agent loop.
- Sections, in this order:
  1. What the project does (3 lines)
  2. Current status: which milestone, what works, what is next
  3. How one question flows through the program, step by step
     (user question -> loop -> model -> tool calls -> results -> answer)
  4. Files and what each one does (one line per file)
  5. Tools: name, input, output, data source
  6. Known limits and open problems
- Rewrite sections in place so the file stays short and current.
  This file describes the present state, not history; history lives in
  DECISIONS.md and git.
- At the end of each session, tell me in one line what you changed in FLOW.md.