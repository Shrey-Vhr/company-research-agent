from langgraph.graph import END, START, StateGraph

from research_agent.graph.nodes import critic, executor, finalize, planner, refuse, writer
from research_agent.graph.state import ResearchState

MAX_REVISIONS = 2


def after_planner(state: ResearchState) -> str:
    return "executor" if state["in_scope"] and state["plan"] else "refuse"


def after_executor(state: ResearchState) -> str:
    return "executor" if state["current_step"] < len(state["plan"]) else "writer"


def after_critic(state: ResearchState) -> str:
    if state["critic_passed"] or state["revision_count"] >= MAX_REVISIONS:
        return "finalize"
    return "writer"


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("writer", writer)
    g.add_node("critic", critic)
    g.add_node("refuse", refuse)
    g.add_node("finalize", finalize)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", after_planner, ["executor", "refuse"])
    g.add_conditional_edges("executor", after_executor, ["executor", "writer"])
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", after_critic, ["finalize", "writer"])
    g.add_edge("refuse", END)
    g.add_edge("finalize", END)
    return g.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke(
        {"question": "Should I buy TCS?", "evidence": []},
        {"recursion_limit": 30},
    )
    print(result["final_report"])