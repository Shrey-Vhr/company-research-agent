# {
#     "step": 1,
#     "tool": "get_financials",
#     "citation": "Yahoo Finance via yfinance, FY2025-26",
#     "content": "...the raw text the tool returned...",
# }

import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    question: str                      # the user's question
    in_scope: bool                     # False for advice or prediction questions
    plan: list[str]                    # research steps written by the planner
    current_step: int                  # which plan step the executor is on
    evidence: Annotated[list[dict], operator.add]  # facts collected by the executor
    draft: str                         # the writer's report
    critique: list[str]                # problems the critic found
    critic_passed: bool                # did the draft pass the critic?
    revision_count: int                # how many times the writer has revised
    human_feedback: str                # your change request, empty if none
    final_report: str                  # what the user sees at the end
    auto_approve: bool                 # True = skip human review (used by evals)