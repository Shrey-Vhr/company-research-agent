import os
import sys
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.messages import HumanMessage, SystemMessage, ToolMessage

from research_agent.tools.financials import get_financials
from research_agent.tools.calculator import calculate
from research_agent.prompts import SYSTEM_PROMPT
from research_agent.tools.reports import search_reports

load_dotenv()
def run_agent(question: str) -> str:
    tools = {"get_financials": get_financials, "calculate": calculate,  "search_reports": search_reports}
    model = ChatOpenAI(model=os.environ["LLM_MODEL"]).bind_tools(
        list(tools.values()), parallel_tool_calls=True
    )
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]
    total_in, total_out = 0, 0
    for round_no in range(1, 9):              # max 8 rounds
        reply = model.invoke(messages)
        usage = reply.usage_metadata or {}
        total_in += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)
        messages.append(reply)

        if not reply.tool_calls:
            print(f"Tokens: {total_in} in, {total_out} out")
            return reply.text

        for call in reply.tool_calls:
            print(f"Round {round_no}: {call['name']}({call['args']})")
            seconds = 0.0
            tool_fn = tools.get(call["name"])
            start = time.perf_counter()
            if tool_fn is None:
                result = f"Error: unknown tool {call['name']}"
            else:
                try:
                    result = tool_fn.invoke(call["args"])
                except Exception as e:
                    result = f"Error: {e}"
            print(f"Round {round_no}: returned {len(str(result))} chars in {seconds:.2f}s: {str(result)[:200]}")
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return "Stopped after 8 rounds without a final answer."

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]).strip()  # all words after the module name, so quotes are optional
    if not question:
        print('Usage: python -m research_agent.raw_agent.loop "What was TCS\'s revenue in FY2025-26?"',
              file=sys.stderr)
        sys.exit(2)  # 2 = command used wrongly (Unix convention)
    print(run_agent(question))