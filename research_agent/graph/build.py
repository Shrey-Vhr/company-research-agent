import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from research_agent.graph.nodes import critic, executor, finalize, planner, refuse, writer, human_review
from research_agent.graph.state import ResearchState

MAX_REVISIONS = 2
CHECKPOINT_DB = Path("data/checkpoints.sqlite")


def after_planner(state: ResearchState) -> str:
    return "executor" if state["in_scope"] and state["plan"] else "refuse"

def after_writer(state: ResearchState) -> str:
    if state.get("skip_critic"):
        return "finalize" if state.get("auto_approve") else "human_review"
    return "critic"


def after_executor(state: ResearchState) -> str:
    return "executor" if state["current_step"] < len(state["plan"]) else "writer"


def after_critic(state: ResearchState) -> str:
    if not state["critic_passed"] and state["revision_count"] < MAX_REVISIONS:
        return "writer"
    if state.get("auto_approve"):
        return "finalize"
    return "human_review"


def after_human(state: ResearchState) -> str:
    return "writer" if state["human_feedback"] else "finalize"


def default_checkpointer():
    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    return SqliteSaver(conn)


def build_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = default_checkpointer()
    g = StateGraph(ResearchState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("writer", writer)
    g.add_node("critic", critic)
    g.add_node("refuse", refuse)
    g.add_node("human_review", human_review)
    g.add_node("finalize", finalize)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", after_planner, ["executor", "refuse"])
    g.add_conditional_edges("executor", after_executor, ["executor", "writer"])
    g.add_conditional_edges("writer", after_writer, ["critic", "human_review", "finalize"])
    g.add_conditional_edges("critic", after_critic, ["writer", "human_review", "finalize"])
    g.add_conditional_edges("human_review", after_human, ["writer", "finalize"])
    g.add_edge("refuse", END)
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


# To view flowchart.. paste the output of the below on mermaid.live
# python -c "from research_agent.graph.build import build_graph; print(build_graph().get_graph().draw_mermaid())"