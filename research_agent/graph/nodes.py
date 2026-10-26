import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from research_agent.graph.state import ResearchState

load_dotenv()

class Plan(BaseModel):
    in_scope: bool = Field(description="False if the question asks for buy/sell/hold advice, price predictions, or anything the tools cannot answer")
    steps: list[str] = Field(description = "2 to 6 short research steps, each one tool action")

PLANNER_PROMPT= """
Your role is to plan research on TCS, INFY, WIPRO using three tools mentioned below:
1. get_financials - Gets yearly financials for an Indian company listed on the NSE.
2. calculate - Evaluates an arithmetic expression and return the result.
3. search_reports - Searches the annual reports of Indian IT companies for facts written in the text.

Steps should be short, and each one should be for a single action. for example “Get TCS financials for 3 years” or “Search the TCS report for headcount”.

Use 2-6 steps and put calculations after the steps that fetch the numbers required. 

Mark the question out of scope if it asks for buy, sell or hold advice, price predictions, or a company other than TCS, INFY or WIPRO. Do not answer the question yourself; only plan.

If the question is out of scope, set in_scope to false and return an empty list of steps.

Data available: financials for up to the last 4 financial years (newest is
FY2025-26); annual reports only for FY2025-26.
"""

# What it does: sends the question to the model, gets back a Plan object, and returns three state fields. The other fields stay untouched.
def planner(state: ResearchState) -> dict:
    llm = ChatOpenAI(model=os.environ["LLM_MODEL"]).with_structured_output(Plan)
    plan = llm.invoke([("system", PLANNER_PROMPT), ("human", state["question"])])
    return {"in_scope": plan.in_scope, "plan": plan.steps, "current_step": 0}

if __name__ == "__main__":
    for q in ["Compare TCS and Infosys net margin in FY2025-26",
              "Who is the CEO of Wipro?",
              "Should I buy TCS?",
              "What will TCS's revenue be in FY2030?"]:
        print(q, "->", planner({"question": q}))
