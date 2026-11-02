import logging
import sys
import time

from dotenv import load_dotenv
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from research_agent.config import RAW_MAX_ROUNDS, llm_model
from research_agent.prompts import SYSTEM_PROMPT
from research_agent.tools.calculator import calculate
from research_agent.tools.financials import get_financials
from research_agent.tools.reports import search_reports

load_dotenv()

logger = logging.getLogger(__name__)


def run_agent(question: str, return_details: bool = False) -> str | tuple[str, dict]:
    tools = {
        "get_financials": get_financials,
        "calculate": calculate,
        "search_reports": search_reports,
    }
    model = ChatOpenAI(model=llm_model()).bind_tools(
        list(tools.values()), parallel_tool_calls=True
    )
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]
    total_in, total_out = 0, 0
    tool_results = []  # only returned when return_details=True
    for round_no in range(1, RAW_MAX_ROUNDS + 1):
        reply = model.invoke(messages)
        usage = reply.usage_metadata or {}
        total_in += usage.get("input_tokens", 0)
        total_out += usage.get("output_tokens", 0)
        messages.append(reply)

        if not reply.tool_calls:
            logger.info("Tokens: %d in, %d out", total_in, total_out)
            if return_details:
                return reply.text, {
                    "tool_results": tool_results,
                    "input_tokens": total_in,
                    "output_tokens": total_out,
                }
            return reply.text

        for call in reply.tool_calls:
            logger.info("Round %d: %s(%s)", round_no, call["name"], call["args"])
            tool_fn = tools.get(call["name"])
            start = time.perf_counter()
            if tool_fn is None:
                result = f"Error: unknown tool {call['name']}"
            else:
                try:
                    result = tool_fn.invoke(call["args"])
                except Exception as e:  # noqa: BLE001 - a tool failure becomes text for the model to read
                    result = f"Error: {e}"
            seconds = time.perf_counter() - start
            logger.info(
                "Round %d: returned %d chars in %.2fs: %s",
                round_no,
                len(str(result)),
                seconds,
                str(result)[:200],
            )
            tool_results.append(
                {"tool": call["name"], "args": call["args"], "content": str(result)}
            )
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    text = f"Stopped after {RAW_MAX_ROUNDS} rounds without a final answer."
    if return_details:
        return text, {
            "tool_results": tool_results,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }
    return text


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    question = " ".join(
        sys.argv[1:]
    ).strip()  # all words after the module name, so quotes are optional
    if not question:
        print(
            'Usage: python -m research_agent.raw_agent.loop "What was TCS\'s revenue in FY2025-26?"',
            file=sys.stderr,
        )
        sys.exit(2)  # 2 = command used wrongly (Unix convention)
    print(run_agent(question))
